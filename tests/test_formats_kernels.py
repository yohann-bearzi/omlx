# SPDX-License-Identifier: Apache-2.0
"""The row skeleton over a format's dot face, with a toy format: rows equal
the dequantized products, and a face that breaks the contract fails to
compile with a message that names the format's own source.
"""

from __future__ import annotations

import logging
import re
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
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

    template <typename XP, int... I>
    static void copy(omlx_ints<I...>, XP x, thread float* xt) {
      ((xt[I] = float(x[I])), ...);
    }
    template <int... I>
    static float dot(omlx_ints<I...>, const device uint8_t* w, thread const float* xt) {
      return (... + ((float(w[I]) - 128.0f) * xt[I]));
    }

    template <typename XP>
    static float load(XP x, thread float* xt) {
      copy(omlx_iota<VPT>(), x, xt);
      return 0;
    }
    static float madd(float acc, Row r, thread const float* xt, float) {
      return acc + dot(omlx_iota<VPT>(), r.w, xt);
    }

    float finish(int r, float sum) const { return sum * s[r]; }

    using Decoded = Row;
    static Decoded decode(Row r) { return r; }
    template <typename A>
    static float apply(float acc, thread const Decoded& d, thread const float* xt, A aux) {
      return madd(acc, d, xt, aux);
    }
  };
};
}  // namespace omlx_toy_KEY
"""


def _toy(key, source=_TOY, vpt=8) -> Dot:
    return Dot(
        key=f"toy_{key}",
        source=source.replace("KEY", key),
        type=f"omlx_toy_{key}::Dot",
        tensors=("scales",),
        vpt=vpt,
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


def test_a_proof_asked_for_again_is_the_one_kept_and_makes_none(monkeypatch):
    """kernels.proof gives the one proof of a key; asked again it makes no
    Proof to drop."""
    kernels._PROOFS.pop(("probe", 1), None)
    first = kernels.proof("probe", 1)
    monkeypatch.setattr(kernels, "Proof", lambda: pytest.fail("made"))
    assert kernels.proof("probe", 1) is first
    kernels._PROOFS.pop(("probe", 1))


class _Missed(dict):
    """A registry whose first two lookups miss, as two threads' that ask
    for a new key at once."""

    misses = 2

    def get(self, key, default=None):
        if self.misses:
            self.misses -= 1
            return default
        return super().get(key, default)


def test_two_threads_that_miss_a_key_at_once_get_one_proof(monkeypatch):
    monkeypatch.setattr(kernels, "_PROOFS", _Missed())
    assert kernels.proof("probe", 2) is kernels.proof("probe", 2)


def test_a_proof_is_shared_by_what_is_compiled_from_the_same_key():
    face = _toy("proof")
    proof = kernels.proof("kernel", face, mx.float32, 512)
    assert proof == kernels.Proof(ran=set(), failed=False) and proof.graphs == {}
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


def test_a_proof_keeps_graphs_until_it_has_too_many_or_a_launch_fails(monkeypatch):
    monkeypatch.setattr(kernels, "MAX_GRAPHS", 3)
    proof = kernels.Proof()
    for signature in "abc":
        proof.keep(signature, signature.upper())
    assert proof.graphs == {"a": "A", "b": "B", "c": "C"}
    # One more drops them all.
    proof.keep("d", "D")
    assert proof.graphs == {"d": "D"}
    proof.keep("e", "E")
    proof.ran.add("launch")
    assert not proof.failed
    proof.fail()
    assert proof.failed and proof.graphs == {} and proof.ran == {"launch"}
    # Nothing is kept on a proof that failed.
    proof.keep("f", "F")
    assert proof.graphs == {}


class _Graph:
    """What a proof keeps, in the place of a function that MLX compiled."""


def test_a_proof_s_graphs_are_those_of_the_thread_that_asks(monkeypatch):
    """MLX keeps what it traces a compiled function into thread by thread:
    a thread keeps, and finds, the graphs it built, and no other thread's."""
    monkeypatch.setattr(kernels, "MAX_GRAPHS", 2)
    proofs, ended = [kernels.Proof()], []

    def keep(name):
        graph = _Graph()
        weakref.finalize(graph, lambda: ended.append((name, threading.get_ident())))
        proofs[0].keep(name, graph)
        return kept()

    def kept():
        return sorted(proofs[0].graphs)

    here = threading.get_ident()
    with ThreadPoolExecutor(max_workers=1) as pool:
        there = pool.submit(threading.get_ident).result()
        assert keep("a") == ["a"]
        assert pool.submit(kept).result() == []
        assert pool.submit(keep, "b").result() == ["b"] and kept() == ["a"]
        # A thread keeps MAX_GRAPHS of a proof: one more drops its own.
        assert pool.submit(keep, "c").result() == ["b", "c"]
        assert pool.submit(keep, "d").result() == ["d"] and kept() == ["a"]
        assert ended == [("b", there), ("c", there)]
        # A failure drops every thread's, on the thread that reports it.
        pool.submit(proofs[0].fail).result()
        assert proofs[0].failed and kept() == [] and pool.submit(kept).result() == []
        assert sorted(ended[2:]) == [("a", there), ("d", there)]
        # Nothing is kept from then on, by either thread.
        assert pool.submit(keep, "e").result() == [] and keep("f") == []
    proofs[0], ended[:] = kernels.Proof(), []
    pool = ThreadPoolExecutor(max_workers=1)
    there = pool.submit(threading.get_ident).result()
    assert pool.submit(keep, "g").result() == ["g"] and keep("h") == ["h"]
    # A thread's graphs end with it, on it; the proof and the others' live on.
    pool.shutdown()
    assert ended == [("g", there)] and kept() == ["h"]
    # A thread that follows it, of the same ident or another, has none.
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(kept).result() == []
        assert pool.submit(keep, "i").result() == ["i"]
        # The proof's end is the end of every thread's, where it ends.
        del proofs[:]
        assert sorted(ended[1:]) == [("h", here), ("i", here)]
    assert "`proof.graphs` is the calling thread's own" in " ".join(_PAGE.split())


def test_a_thread_keeps_64_graphs_of_a_proof_and_one_more_drops_them():
    """``MAX_GRAPHS`` as it ships, which each thread counts for itself."""
    proof = kernels.Proof()

    def keep(first, count):
        for signature in range(first, first + count):
            proof.keep(signature, _Graph())
        return sorted(proof.graphs)

    assert keep(0, 64) == list(range(64))
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(keep, 100, 64).result() == list(range(100, 164))
        assert sorted(proof.graphs) == list(range(64))
        assert keep(64, 1) == [64]
        assert pool.submit(keep, 164, 0).result() == list(range(100, 164))


def test_no_graph_is_kept_while_a_failure_is_recorded(monkeypatch):
    """A launch can fail on one thread while another builds one: keeping a
    graph (the look at the failure, then the graph) and recording a failure
    (the failure, then the graphs' end) each wait for the other."""
    proof, lock = kernels.Proof(), kernels._KEEPING
    keeping = threading.Thread(target=proof.keep, args=("a", _Graph()))
    failing = threading.Thread(target=proof.fail)
    with lock:
        for thread in (keeping, failing):
            thread.start()
            thread.join(0.1)
            assert thread.is_alive()
        assert not proof.failed
    keeping.join()
    failing.join()
    assert proof.failed and proof.graphs == {}

    # A graph that waits to be kept while a failure is recorded is not kept:
    # the look at the failure waits too.
    class FailedMeanwhile:
        def __enter__(self):
            monkeypatch.setattr(kernels, "_KEEPING", lock)
            proof.fail()
            lock.acquire()

        def __exit__(self, *error):
            lock.release()

    proof = kernels.Proof()
    monkeypatch.setattr(kernels, "_KEEPING", FailedMeanwhile())
    proof.keep("b", _Graph())
    assert proof.failed and proof.graphs == {}


def test_a_graph_kept_while_a_failure_waits_ends_with_it(monkeypatch):
    """The other order: a graph is kept as a failure reaches for the lock.
    The failure and the graphs' end are one step for a thread that keeps."""
    lock = kernels._KEEPING

    class KeptMeanwhile:
        def __enter__(self):
            monkeypatch.setattr(kernels, "_KEEPING", lock)
            proof.keep("a", _Graph())
            lock.acquire()

        def __exit__(self, *error):
            lock.release()

    proof = kernels.Proof()
    monkeypatch.setattr(kernels, "_KEEPING", KeptMeanwhile())
    proof.fail()
    assert proof.failed and proof.graphs == {}


def test_a_thread_finds_its_graphs_without_the_lock(monkeypatch):
    """A decode step's lookup waits for no thread that keeps a graph, or
    that ends as many as it kept."""
    proof = kernels.Proof()
    proof.keep("a", "A")

    class Taken:
        def __enter__(self):
            pytest.fail("the lookup took the lock")

        def __exit__(self, *error):
            pass

    monkeypatch.setattr(kernels, "_KEEPING", Taken())
    assert proof.graphs.get("a") == "A"


def test_graphs_end_once_the_lock_is_released(monkeypatch):
    """A compiled function's end is MLX's work, for as many graphs as a
    thread kept, or as every thread did: nothing waits for it behind the
    lock."""
    monkeypatch.setattr(kernels, "MAX_GRAPHS", 2)
    proof, ended = kernels.Proof(), []

    def keep(name):
        graph = _Graph()
        weakref.finalize(graph, lambda: ended.append((name, kernels._KEEPING.locked())))
        proof.keep(name, graph)

    for name in "abc":
        keep(name)
    # The third dropped the two that were kept; a failure drops the third.
    assert ended == [("a", False), ("b", False)]
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(keep, "d").result()
        proof.fail()
        assert sorted(ended[2:]) == [("c", False), ("d", False)]


_LOG = logging.getLogger("omlx.formats.a_kernel")


def _said(caplog):
    """What a proof logged for its kernel: each record's level, sentence
    and error."""
    return [
        (r.levelname, r.getMessage(), r.exc_info and r.exc_info[1])
        for r in caplog.records
        if r.name == _LOG.name
    ]


def _proves(proof, metal, build):
    return proof.prove(metal, build, _LOG, "engaged", "failed; fallback")


def test_a_proof_evaluates_the_launch_it_is_given_once_for_a_metal(monkeypatch, caplog):
    """A kernel hands a proof the smallest launch of a Metal, as its inputs
    and the function that launches it over them: the proof evaluates the
    inputs, then the launch, and marks the Metal, in the kernel's words."""
    caplog.set_level(logging.INFO, logger=_LOG.name)
    proof, built, launched, evaluated = kernels.Proof(), [], [], []
    evaluate = mx.eval

    def recorded(*arrays):
        evaluated.append(arrays)
        evaluate(*arrays)

    def launch(x):
        launched.append(x + 1)
        return launched[-1]

    def build():
        built.append(mx.zeros((8,)) + len(built))
        return [built[-1]], launch

    monkeypatch.setattr(mx, "eval", recorded)
    assert _proves(proof, "metal", build) is True
    assert proof.ran == {"metal"} and not proof.failed
    assert len(built) == 1 and evaluated == [(built[0],), (launched[0],)]
    assert _said(caplog) == [("INFO", "engaged", None)]
    # A Metal that ran is neither built nor evaluated again. Another one is.
    assert _proves(proof, "metal", build) is True
    assert len(built) == 1 and len(evaluated) == 2 and len(_said(caplog)) == 1
    assert _proves(proof, "other", build) is True
    assert proof.ran == {"metal", "other"}
    assert evaluated[2:] == [(built[1],), (launched[1],)]
    assert len(_said(caplog)) == 2
    # A proof keeps no graph by itself.
    assert proof.graphs == {}


class _ProofError(Exception):
    """An error of a type that is no other's."""


def _raise(error):
    raise error


def test_a_launch_that_fails_its_evaluation_fails_the_proof(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=_LOG.name)
    proof, error, built = kernels.Proof(), _ProofError("not evaluated"), []
    proof.ran.add("ran")
    proof.keep("signature", _Graph())

    def build():
        built.append(mx.zeros((8,)))
        return [], lambda: built[-1]

    with monkeypatch.context() as patch:
        # The launch's evaluation fails: the stand-in has no input.
        patch.setattr(mx, "eval", lambda *arrays: arrays and _raise(error))
        assert _proves(proof, "metal", build) is False
    assert proof.failed and proof.ran == {"ran"} and proof.graphs == {}
    assert _said(caplog) == [("WARNING", "failed; fallback", error)]
    # Nothing is built for a proof that failed: for another Metal, or for
    # one that ran.
    assert _proves(proof, "other", build) is False
    assert _proves(proof, "ran", build) is False
    assert len(built) == 1 and len(_said(caplog)) == 1


def test_what_fails_as_a_proof_s_launch_is_built_is_raised(caplog):
    """A launch that cannot be built is not one that failed: the error is
    the kernel's own, or its caller's, whether the stand-in or the launch
    over it raised it."""
    caplog.set_level(logging.INFO, logger=_LOG.name)
    proof, error = kernels.Proof(), _ProofError("no launch")
    for build in (
        lambda: _raise(error),
        lambda: ([mx.zeros((8,))], lambda x: _raise(error)),
    ):
        with pytest.raises(_ProofError) as raised:
            _proves(proof, "metal", build)
        assert raised.value is error
    assert not proof.failed and not proof.ran and not _said(caplog)


def test_a_stand_in_left_lazy_is_evaluated_by_the_proof_ahead_of_its_launch(
    monkeypatch,
):
    """A kernel's stand-in hands its inputs to the proof unevaluated: the
    proof evaluates them and synchronizes the GPU's stream, outside the
    guard, then builds the launch over them and evaluates it."""
    proof, events, given, launched = kernels.Proof(), [], [], []
    evaluate, synchronize = mx.eval, mx.synchronize
    lazy = mx.add(mx.zeros((8,), stream=mx.gpu), 1, stream=mx.gpu)

    def evaluated(*arrays):
        events.append("eval")
        given.append(arrays)
        evaluate(*arrays)

    def synchronized(*stream):
        events.append(("synchronize", *stream))
        synchronize(*stream)

    def launch(x):
        events.append("launch")
        given.append((x,))
        launched.append(x * 2)
        return launched[-1]

    with monkeypatch.context() as patch:
        patch.setattr(mx, "eval", evaluated)
        patch.setattr(mx, "synchronize", synchronized)
        assert _proves(proof, "metal", lambda: ([lazy], launch)) is True
    assert events == ["eval", ("synchronize", mx.gpu), "launch", "eval"]
    assert [[id(a) for a in arrays] for arrays in given] == [
        [id(lazy)],
        [id(lazy)],
        [id(launched[0])],
    ]
    assert mx.array_equal(launched[0], mx.full((8,), 2.0)).item()
    assert proof.ran == {"metal"} and not proof.failed


def test_a_stand_in_s_inputs_may_be_given_once_through(caplog):
    """The proof reads the inputs a stand-in gives once: it evaluates them,
    then launches over them, whatever iterable holds them."""
    proof = kernels.Proof()
    inputs = (a for a in [mx.zeros((8,)), mx.ones((8,))])
    assert _proves(proof, "metal", lambda: (inputs, lambda x, y: x + y)) is True


def test_an_allocation_that_fails_in_a_stand_in_is_raised_and_proven_again(caplog):
    """MLX allocates a stand-in's zeros when the proof evaluates them, ahead
    of the guard: an allocation that fails is raised, the proof has neither
    failed nor proven the Metal, and the next call proves it."""
    caplog.set_level(logging.INFO, logger=_LOG.name)
    proof, launched = kernels.Proof(), []

    def launch(x):
        launched.append(x + 1)
        return launched[-1]

    # 2**60 bytes: past any device's largest buffer.
    huge = kernels.zeros(((2**30, 2**30), mx.uint8))
    with pytest.raises(RuntimeError, match="allocate"):
        _proves(proof, "metal", lambda: (huge, launch))
    assert not launched and not proof.failed and not proof.ran
    assert not _said(caplog)
    stand_in = kernels.zeros(((8,), mx.float32))
    assert _proves(proof, "metal", lambda: (stand_in, launch)) is True
    assert len(launched) == 1 and proof.ran == {"metal"} and not proof.failed
    assert _said(caplog) == [("INFO", "engaged", None)]


def test_two_threads_prove_a_metal_at_once_without_waiting_for_each_other(
    monkeypatch,
):
    """No lock is held while a launch is built, nor while it is evaluated:
    each of the threads that meet a Metal unproven proves it, and it is
    marked once."""
    proof, together = kernels.Proof(), threading.Barrier(2, timeout=10)
    evaluate, met = mx.eval, []

    def build():
        met.append(("built", threading.get_ident()))
        together.wait()
        return [], lambda: mx.zeros((8,))

    def evaluated(*arrays):
        if arrays:  # the launch: the stand-in has no input
            met.append(("evaluated", threading.get_ident()))
            together.wait()
        evaluate(*arrays)

    monkeypatch.setattr(mx, "eval", evaluated)
    with ThreadPoolExecutor(max_workers=2) as pool:
        proving = [pool.submit(_proves, proof, "metal", build) for _ in range(2)]
        assert [proved.result(timeout=30) for proved in proving] == [True, True]
    # Both threads were in the build together, then in the evaluation.
    assert [step for step, _ in met] == ["built"] * 2 + ["evaluated"] * 2
    assert len({thread for _, thread in met}) == 2
    assert proof.ran == {"metal"} and not proof.failed


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
    # Five #line (the index packs, the source, the revisions and the checks,
    # the row skeleton, the kernel's header), the two revisions and their two
    # ends, each on a line of its own.
    assert len(marked) == 9 and all(line.startswith("#") for line in marked)


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


@metal
@pytest.mark.parametrize("packs", [(1, 0), (2, 2), (3, 1)])
@pytest.mark.parametrize("k", [512, 384, 248, 8])
def test_a_face_loads_inputs_staged_in_threadgroup_memory(k, packs):
    """A kernel may copy its inputs into threadgroup memory and run the
    skeleton over them there: the face's load takes either pointer."""
    operands, x, want = _matrix(16, k, seed=k + sum(packs))
    face = _toy("staged")
    got = testing.dot_rows(face, operands, x, 16, packs, staged=True)
    assert mx.array_equal(got, want).item()


@metal
def test_a_load_of_device_inputs_alone_fails_on_staged_ones_naming_load():
    device_only = _TOY.replace(
        "    template <typename XP>\n    static float load(XP x, thread float* xt) {",
        "    template <typename X>\n"
        "    static float load(const device X* x, thread float* xt) {",
    )
    assert device_only != _TOY
    face = _toy("device_only", source=device_only)
    operands, x, want = _matrix(8, 512)
    assert mx.array_equal(testing.dot_rows(face, operands, x, 8), want[:8]).item()
    with pytest.raises(RuntimeError) as failure:
        mx.eval(testing.dot_rows(face, operands, x, 8, staged=True))
    assert "no matching function for call to 'load'" in str(failure.value)


@metal
def test_the_skeleton_s_inputs_are_of_the_type_it_is_given():
    """qmv_rows<T, ...> takes x as T inputs, in device or threadgroup memory:
    inputs of another type do not compile."""
    face = _toy("typed")
    matrices = kernels.Matrices()
    kernel = mx.fast.metal_kernel(
        name="omlx_test_typed_inputs",
        input_names=["x", *matrices.add("m", face)],
        output_names=["y"],
        header=matrices.header + kernels.STORE_ROWS,
        source=matrices.declarations + r"""
    float result[1];
    qmv_rows<half, K, 1, 0>(m, 0, m, 0, x, thread_index_in_simdgroup, result);
    y[0] = result[0];
""",
    )
    (codes, scales), x, _ = _matrix(8, 512)
    with pytest.raises(RuntimeError) as failure:
        mx.eval(
            kernel(
                inputs=[x, codes, scales],
                template=[("K", 512)],
                grid=(32, 1, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[(1,)],
                output_dtypes=[mx.float32],
            )[0]
        )
    assert "no matching function for call to 'omlx_inputs'" in str(failure.value)


@metal
def test_staged_inputs_may_fill_a_threadgroup_s_memory():
    operands, x, want = _matrix(8, 8192)
    assert x.nbytes == 32768
    got = testing.dot_rows(_toy("full"), operands, x, 8, staged=True)
    assert mx.array_equal(got, want).item()


def _vectors(rows, k, seed=0):
    """``rows`` vectors of small integer activations, and each one's toy
    products with ``_matrix(16, k, seed)``'s rows."""
    operands, _, _ = _matrix(16, k, seed)
    codes, scales = operands
    mx.random.seed(seed + 1000)
    xs = mx.random.randint(-8, 9, (rows, k)).astype(mx.float32)
    want = (xs @ (codes.astype(mx.float32) - 128.0).T) * scales
    return operands, xs, want


@metal
@pytest.mark.parametrize("k", [512, 384, 248, 8])
@pytest.mark.parametrize(
    "rows, rps",
    [(1, 1), (3, 2), (16, 4), (5, 4), (16, 1)],
)
def test_the_shared_rows_skeleton_gives_each_vector_its_rows(k, rows, rps):
    """Each vector's products with the rows, every block's weights decoded
    once for all the vectors: the toy's at K of whole blocks, of a last
    block of 16 or 31 lanes, of one lane; packs of 1, 2, 4 rows; N not
    whole packs (10 rows of packs of 4: the last pack holds 2)."""
    operands, xs, want = _vectors(rows, k, seed=k + rows)
    face = _toy("shared_rows")
    for n in (16, 12, 10):
        got = testing.shared_rows(face, operands, xs, n, rps=rps)
        assert mx.array_equal(got, want[:, :n]).item()


@metal
def test_an_unrolled_shared_rows_skeleton_gives_the_same_rows():
    operands, xs, want = _vectors(4, 512)
    face = _toy("unrolled")
    for rps in (1, 2, 4):
        got = testing.shared_rows(face, operands, xs, 16, rps=rps, unrolled=True)
        assert mx.array_equal(got, want).item()


def test_a_shared_rows_launch_is_sized_by_the_face_s_vpt():
    # Fewer than 2048 tiles of 8 rows: one vector a threadgroup.
    assert kernels.shared_geometry(8, 16376, 6, pad=False) == (1, 4)
    # From 2048: as many vectors as divide the count, within 32 values a lane.
    assert kernels.shared_geometry(8, 16384, 6, pad=False) == (3, 4)
    assert kernels.shared_geometry(16, 16384, 6, pad=False) == (2, 4)
    assert kernels.shared_geometry(4, 16384, 16, pad=False) == (8, 4)
    # A count no divisor groups stays one a group, or is padded to the budget.
    assert kernels.shared_geometry(8, 16384, 7, pad=False) == (1, 4)
    assert kernels.shared_geometry(8, 16384, 7, pad=True) == (4, 4)
    assert kernels.shared_geometry(8, 16384, 2, pad=True) == (2, 4)
    # The unrolled envelope: 16 accumulators, 80 values a lane.
    assert kernels.unrolled_tile_ok(16, 4, 4) and not kernels.unrolled_tile_ok(16, 4, 5)
    assert kernels.unrolled_tile_ok(16, 8, 2) and not kernels.unrolled_tile_ok(16, 8, 3)
    assert kernels.unrolled_tile_ok(20, 1, 4) and not kernels.unrolled_tile_ok(16, 1, 6)


def test_a_kernel_carries_the_skeletons_it_calls_and_no_other():
    """MLX hashes a kernel's whole source at every launch: a kernel whose
    matrices all share has the checks and the shared-rows skeleton, and no
    row skeleton; one that reads a matrix plainly too has both. Each
    skeleton stays at its own lines."""
    face = _toy("carried")
    shared, both = kernels.Matrices(), kernels.Matrices()
    shared.add("m", face, shared=True)
    both.add("m", face, shared=True)
    both.add("n", face)
    assert "void qmv_rows(" not in shared.header and "void qmv_shared(" in shared.header
    assert "void qmv_rows(" in both.header and "void qmv_shared(" in both.header
    assert "struct omlx_revision" in shared.header
    lines = (kernels._revisions() + kernels._CHECKS + kernels._QMV_ROWS).count("\n")
    start = (
        kernels._INDEX_PACKS.count("\n") + 1 + lines + kernels._TILE_FACE.count("\n")
    )
    for header in (shared.header, both.header):
        assert f'\n#line {start} "{kernels._SKELETONS}"\n' in header


def test_the_skeletons_are_carried_compact():
    """No indentation and no comment line in the Metal a kernel carries; the
    lines stay, so that the compiler cites them where the source has them."""
    for skeleton in (
        kernels._INDEX_PACKS,
        kernels._CHECKS,
        kernels._QMV_ROWS,
        kernels._TILE_FACE,
        kernels._QMV_SHARED,
        kernels.STORE_ROWS,
    ):
        lines = skeleton.split("\n")
        assert all(line == line.strip() for line in lines)
        assert not any(line.startswith("//") for line in lines)
    assert kernels._compact("  a;\n  // b\n    c; // d\n") == "a;\n\nc; // d\n"


def test_a_kernel_that_shares_unrolls_its_skeleton_when_asked():
    face = _toy("unroll")
    rolled, unrolled = kernels.Matrices(), kernels.Matrices(unrolled=True)
    rolled.add("m", face, shared=True)
    unrolled.add("m", face, shared=True)
    pragma = '_Pragma("clang loop unroll(full)")'
    assert pragma not in rolled.header and pragma in unrolled.header
    # The skeleton's block walk stays a loop.
    assert "\nfor (int k = 0; k < WHOLE; k += L::BLOCK) {\n" in unrolled.header


def test_shared_rows_launch_a_partial_pack_where_n_is_not_whole_packs(monkeypatch):
    operands, xs, _ = _vectors(2, 512)
    face = _toy("partial")
    kernel, asked = testing._shared_rows_kernel(face, False), []

    def recorded(**named):
        asked.append(dict(named["template"]))
        return kernel(**named)

    monkeypatch.setattr(testing, "_shared_rows_kernel", lambda *key: recorded)
    for n, rps in ((16, 4), (10, 4), (10, 2), (7, 1)):
        testing.shared_rows(face, operands, xs, n, rps=rps)
    assert [(t["N"], t["RPS"], t["PARTIAL"]) for t in asked] == [
        (16, 4, False),
        (10, 4, True),
        (10, 2, False),
        (7, 1, False),
    ]


def test_shared_rows_take_1_to_16_vectors_of_a_face_with_a_vpt():
    operands, xs, _ = _vectors(17, 512)
    with pytest.raises(ValueError, match="1 to 16 vectors"):
        testing.shared_rows(_toy("many"), operands, xs, 16)
    with pytest.raises(ValueError, match="1 to 16 vectors"):
        testing.shared_rows(_toy("many"), operands, xs[0], 16)
    with pytest.raises(ValueError, match="states no vpt"):
        testing.shared_rows(_toy("no_vpt", vpt=0), operands, xs[:2], 16)


@metal
def test_a_vpt_that_is_not_the_struct_s_is_refused_in_words():
    operands, xs, _ = _vectors(2, 512)
    with pytest.raises(RuntimeError) as failure:
        mx.eval(testing.shared_rows(_toy("vpt4", vpt=4), operands, xs, 16))
    assert "dot face: Lanes<K>::VPT must be the face's vpt" in str(failure.value)


@metal
def test_an_apply_that_is_not_madd_is_told_apart():
    """The rule a format's tests check: shared_rows against dot_rows."""
    other = _TOY.replace(
        "      return madd(acc, d, xt, aux);",
        "      return madd(acc, d, xt, aux) + 1.0f;",
    )
    assert other != _TOY
    face = _toy("other_apply", source=other)
    operands, x, want = _matrix(16, 512)
    assert mx.array_equal(testing.dot_rows(face, operands, x, 16), want).item()
    got = testing.shared_rows(face, operands, x[None], 16)[0]
    assert not mx.array_equal(got, want).item()


@metal
def test_a_face_without_decode_is_told_which_member_it_lacks():
    decode = "    static Decoded decode(Row r) { return r; }\n"
    assert decode in _TOY
    face = _toy("no_decode", source=_TOY.replace(decode, ""))
    operands, x, want = _matrix(8, 512)
    # The row skeleton needs none: madd is the row's product there.
    assert mx.array_equal(testing.dot_rows(face, operands, x, 8), want).item()
    with pytest.raises(RuntimeError) as failure:
        mx.eval(testing.shared_rows(face, operands, x[None], 8))
    assert "no member named 'decode'" in str(failure.value)


def test_a_face_s_shared_source_is_in_the_kernels_that_share_alone():
    face = replace(_toy("shared"), shared="// the shared source\n")
    with pytest.raises(ValueError, match="states no vpt"):
        kernels.Matrices().add("m", replace(face, vpt=0), shared=True)
    plain = kernels.Matrices()
    plain.add("m", face)
    assert "// the shared source" not in plain.header
    both = kernels.Matrices()
    both.add("a", face, shared=True)
    both.add("b", face, shared=True)
    header = both.header
    assert header.count("// the shared source") == 1
    # After the source whose struct it defines members of.
    assert header.index("// the shared source") > header.index("struct Dot {")
    assert '#line 1 "<quant format toy_shared shared>"' in header
    tile = formats.Tile(key="t", source="", type="T", tensors=())
    with pytest.raises(TypeError, match="a tile face has no shared source"):
        kernels.Matrices().add("t", tile, shared=True)
    with pytest.raises(ValueError, match="shared is Metal"):
        replace(face, shared=3)
    for vpt in (-1, 1.0, True):
        with pytest.raises(ValueError, match="vpt is a count of inputs"):
            replace(face, vpt=vpt)
    # Only a kernel that shares includes the skeleton that does.
    assert "qmv_shared" not in plain.header and "qmv_shared" in header


def test_staged_inputs_fit_in_a_threadgroup_s_memory():
    operands, _, _ = _matrix(8, 512)
    x = mx.zeros((8192 + 8,), mx.float32)
    with pytest.raises(ValueError, match="32768 bytes of a threadgroup's memory"):
        testing.dot_rows(_toy("big"), operands, x, 8, staged=True)


# The toy's rows with the codes as stored, re-centred in madd with the sum of
# the lane's inputs: what load returns is a struct of the face's own.
_LOAD = r"""
    template <typename XP>
    static float load(XP x, thread float* xt) {
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
    template <typename XP>
    static Inputs load(XP x, thread float* xt) {
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


@metal
@pytest.mark.parametrize("block", ["32 * VPT", "256"])
def test_a_lane_of_no_inputs_is_refused_in_words_and_nothing_else(block):
    source = _TOY.replace("VPT = 8;", "VPT = 0;").replace("32 * VPT", block)
    message = _error(_toy(f"no_inputs_{block[:2]}", source=source))
    assert (
        "due to requirement 'Lanes<512>::VPT >= 1' "
        '"dot face: Lanes<K>::VPT must be one input or more"'
    ) in " ".join(message.split())
    assert "<omlx skeletons, docs/quant-formats.md>:" in message
    # Nothing that multiplies or divides by VPT or walks a lane's inputs is
    # compiled.
    assert message.count("error:") == 1


@metal
@pytest.mark.parametrize("k", [32, 20])
def test_a_lane_of_one_input_computes_its_rows(k):
    """The fewest inputs a lane may read: whole blocks, and a last block of
    20 lanes."""
    face = _toy("one_input", source=_TOY.replace("VPT = 8;", "VPT = 1;"))
    operands, x, want = _matrix(8, k)
    assert mx.array_equal(testing.dot_rows(face, operands, x, 8), want).item()


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
    # A face that names no dtype is exact for the floats.
    floats = "toy_a is exact for mlx.core.float16, mlx.core.bfloat16, mlx.core.float32"
    with pytest.raises(ValueError, match=floats):
        testing.dot_rows(_toy("a"), operands, x.astype(mx.int32), 8)
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


def _counted(monkeypatch):
    """The kernels mx.fast.metal_kernel builds, and the Matrices made."""
    built, made = [], []
    real_kernel, real_matrices = mx.fast.metal_kernel, kernels.Matrices

    def kernel(**kwargs):
        built.append(kwargs)
        return real_kernel(**kwargs)

    def matrices():
        made.append(1)
        return real_matrices()

    monkeypatch.setattr(mx.fast, "metal_kernel", kernel)
    monkeypatch.setattr(kernels, "Matrices", matrices)
    # A registry of this test's own: what another test built is not found.
    monkeypatch.setattr(kernels, "_FACE_KERNELS", {})
    return built, made


def test_a_kernel_over_faces_is_written_once(monkeypatch):
    built, made = _counted(monkeypatch)
    face = _toy("written_once")
    args = (
        "two_rows",
        ("x", "m", "rows"),
        {"m": face},
        ("y",),
        "// header\n",
        "y[0] = 0;",
    )
    first = kernels.face_kernel(*args)
    assert kernels.face_kernel(*args) is first
    assert (len(built), len(made)) == (1, 1)
    (kwargs,) = built
    assert kwargs["name"] == f"two_rows_{face.key}"
    assert kwargs["input_names"] == [
        "x",
        "m_weight",
        *(f"m_{t}" for t in face.tensors),
        "rows",
    ]
    assert kwargs["output_names"] == ["y"]
    expected = kernels.Matrices()
    expected.add("m", face)
    assert kwargs["header"] == expected.header + "// header\n"
    assert kwargs["source"] == expected.declarations + "y[0] = 0;"


@pytest.mark.parametrize(
    "change", ["name", "inputs", "face", "outputs", "header", "source"]
)
def test_a_kernel_over_faces_is_written_again_for_anything_else(monkeypatch, change):
    built, _ = _counted(monkeypatch)
    face, other = _toy("written_a"), _toy("written_b")
    args = {
        "name": "two_rows",
        "inputs": ("x", "m"),
        "matrices": {"m": face},
        "outputs": ("y",),
        "header": "",
        "source": "y[0] = 0;",
    }
    kernels.face_kernel(**args)
    args.update(
        {
            "name": {"name": "three_rows"},
            "inputs": {"inputs": ("m", "x")},
            "face": {"matrices": {"m": other}},
            "outputs": {"outputs": ("z",)},
            "header": {"header": "// another\n"},
            "source": {"source": "y[1] = 0;"},
        }[change]
    )
    kernels.face_kernel(**args)
    assert len(built) == 2


def test_a_kernel_over_faces_reads_every_matrix_it_names():
    with pytest.raises(ValueError, match="matrices m are not among its inputs"):
        kernels.face_kernel("two_rows", ("x",), {"m": _toy("unread")}, ("y",), "", "")
    unread = {"b": _toy("unread_b"), "a": _toy("unread_a")}
    with pytest.raises(ValueError, match="matrices a, b are not among its inputs"):
        kernels.face_kernel("two_rows", ("x",), unread, ("y",), "", "")


def test_a_kernel_over_two_faces_takes_them_in_the_order_of_its_inputs(
    monkeypatch,
):
    built, _ = _counted(monkeypatch)
    fa, fb = _toy("order_a"), _toy("order_b")
    inputs = ["x", "b", "a"]  # any sequence of names
    first = kernels.face_kernel("pair", inputs, {"a": fa, "b": fb}, ["y"], "", "")
    again = kernels.face_kernel(
        "pair", tuple(inputs), {"b": fb, "a": fa}, ("y",), "", ""
    )
    assert first is again and len(built) == 1
    (kwargs,) = built
    assert kwargs["name"] == f"pair_{fb.key}_{fa.key}"
    assert kwargs["input_names"] == [
        "x",
        "b_weight",
        *(f"b_{t}" for t in fb.tensors),
        "a_weight",
        *(f"a_{t}" for t in fa.tensors),
    ]
    expected = kernels.Matrices()
    expected.add("b", fb)
    expected.add("a", fa)
    assert kwargs["header"] == expected.header
    # Another matrix under each name is another kernel.
    kernels.face_kernel("pair", inputs, {"a": fb, "b": fa}, ["y"], "", "")
    assert len(built) == 2


@pytest.mark.parametrize("which", ["inputs", "outputs"])
def test_a_kernel_over_faces_takes_no_string_of_names(which):
    args = {"inputs": ("x", "m"), "outputs": ("y",)}
    args[which] = "xm"
    with pytest.raises(TypeError, match="sequences of names, not a string"):
        kernels.face_kernel(
            "two_rows", args["inputs"], {"m": _toy("string")}, args["outputs"], "", ""
        )
