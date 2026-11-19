# SPDX-License-Identifier: Apache-2.0
"""Checks for a format's tests. ``dot_rows`` runs a face in the skeletons
oMLX's kernels are built from, outside any model, ``shared_rows`` runs
several vectors through ``decode`` and ``apply``, ``lane_parts`` reads each
lane's part of rows through cursors stepped and placed, ``finish_rows``
gives ``finish`` sums to scale, ``rows_stack`` checks the rows rule's
promise of a format, and ``tile_rows`` reads every weight of rows through
a tile face. ``staged_tables`` reads a face's tables back as a kernel
stages them. ``no_dot_defects`` fails a test in
which a format's ``dot()`` or ``tile()`` was defective. ``layer_calls`` records
where format layers' rows ran: their face's launch, or the format's decode.
``fresh_proofs`` gives a test proofs of its own. ``verified`` has oQ take model families
as verified with quant formats for a test's checkpoints.

Nothing here is imported by the layers or the kernels: what a check needs of
them it wraps for the length of a block.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache, wraps
from typing import Any

import mlx.core as mx

from .. import formats
from . import Dot, Tile, kernels
from .linear import FormatLinear

# One threadgroup of one simdgroup per NA + NB rows. The rows read x where
# the kernel has it (XS): in device memory, or staged in threadgroup memory.
_ROWS = r"""
    const uint simd_lid = thread_index_in_simdgroup;
    const size_t row = size_t(threadgroup_position_in_grid.y) * (NA + NB);
    float result[NA + NB];
    qmv_rows<T, K, NA, NB>(m, row, m, row + NA, XS, simd_lid, result);
    if (simd_lid == 0) {
      store_rows(omlx_iota<NA + NB>(), result, y + row);
    }
"""

# x copied into threadgroup memory, as a kernel that stages its inputs does.
_STAGED = r"""
    threadgroup T xs[K];
    for (int i = int(thread_index_in_simdgroup); i < K; i += 32) {
      xs[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

# The shared-rows skeleton over ROWS inputs: RPS output rows a simdgroup, the
# last pack of `valid` rows where N is not whole packs.
_SHARED_ROWS = r"""
    const size_t row = size_t(threadgroup_position_in_grid.y) * RPS;
    const int valid = min(RPS, N - int(row));
    float result[ROWS * RPS];
    qmv_shared<T, K, ROWS, RPS, VPT, PARTIAL>(
        m, row, valid, x, thread_index_in_simdgroup, result);
    if (thread_index_in_simdgroup == 0) {
      for (int i = 0; i < ROWS; i++) {
        for (int r = 0; r < valid; r++) {
          y[i * N + row + r] = static_cast<T>(result[i * RPS + r]);
        }
      }
    }
"""

# MLX's threadgroup memory for a kernel, in bytes.
_THREADGROUP_BYTES = 32768


# The families of the open blocks of verified, in any thread.
_VERIFYING: list[frozenset[str]] = []
_VERIFYING_LOCK = threading.Lock()


def verified(*families: str) -> _Verified:
    """While a block of it is open, in any thread, oQ writes and oMLX loads
    checkpoints of ``families`` in a quant format as it does those verified
    with them. Name the top-level and nested text model types. A test can
    also construct format layers by hand within this block. Loader admission
    still applies to embedding, reranking and other dedicated readers.
    The decorator covers the whole test, including async tests. In a
    fixture, use ``with verified(...): yield``.
    Blocks of several threads open and close in any order."""
    return _Verified(*families)


class _Verified:
    """A block of ``verified``, and the decorator it is."""

    def __init__(self, *families: str):
        if not families or not all(
            isinstance(family, str) and family.strip() for family in families
        ):
            raise TypeError(
                "testing.verified takes the model types it verifies: "
                "@verified('qwen3_5_moe_text'), not a bare @verified"
            )
        from .. import oq

        self._families = frozenset(oq._family(family) for family in families)

    def __enter__(self):
        with _VERIFYING_LOCK:
            _VERIFYING.append(self._families)
            _publish()
        return self

    def __exit__(self, *raised):
        with _VERIFYING_LOCK:
            _VERIFYING.remove(self._families)
            _publish()

    def __call__(self, test):
        if not inspect.isfunction(test) or hasattr(test, "_pytestfixturefunction"):
            raise TypeError(
                "testing.verified decorates a test function: a class's tests "
                "and a fixture open a block of it (with verified(...):)"
            )
        if inspect.isgeneratorfunction(test) or inspect.isasyncgenfunction(test):
            raise TypeError(
                "testing.verified decorates a test; a fixture opens a block of "
                "it: with verified(...): yield"
            )
        if inspect.iscoroutinefunction(test):

            @wraps(test)
            async def run(*args, **kwargs):
                with self:
                    return await test(*args, **kwargs)

        else:

            @wraps(test)
            def run(*args, **kwargs):
                with self:
                    return test(*args, **kwargs)

        return run


def _publish() -> None:
    """oQ's test-verified families: those of the open blocks."""
    from .. import oq

    oq._TEST_VERIFIED = frozenset().union(*_VERIFYING)
    formats._TESTING = bool(_VERIFYING)


@contextmanager
def no_dot_defects():
    """Fail on leaving the block when a format's ``dot()`` or ``tile()`` was
    defective in it.

    ``omlx.formats.dot`` and ``omlx.formats.tile`` log a defect and answer
    None: a model keeps its other kernels. A test wants the failure. oMLX
    runs each test under ``tests/`` in this block, and so can a plugin, from
    an autouse fixture. An exception of the block's body is raised in place
    of this failure.

    The block's value is the set of the defects met so far, each a format's
    mode and the error's ``repr``. A test that expects one removes it.

    The set is the process's while the block is open: the latest block still
    open takes the defects that any thread meets, and a block opened inside
    another (in any thread) takes them from the other until it ends. Blocks
    may end in any order, in any thread, as those of ``fresh_proofs`` do."""
    with _DEFECTS.block(set()) as defects:
        yield defects
    if defects:
        raise AssertionError(
            f"a quant format's dot() or tile() is defective: {sorted(defects)}"
        )


@cache
def _dot_rows_kernel(face: Dot, staged: bool):
    matrices = kernels.Matrices()
    inputs = ["x", *matrices.add("m", face)]
    rows = _ROWS.replace("XS", "(const threadgroup T*)xs" if staged else "x")
    return mx.fast.metal_kernel(
        name=f"omlx_format_dot_rows_{face.key}",
        input_names=inputs,
        output_names=["y"],
        header=matrices.header + kernels.STORE_ROWS,
        source=matrices.declarations + (_STAGED if staged else "") + rows,
    )


@cache
def _staged_tables_kernel(face: Dot | Tile):
    matrices = kernels.Matrices()
    inputs = matrices.add("m", face)
    copies = "".join(
        f"      for (uint i = 0; i < {len(table.values)}u; ++i) {{\n"
        f"        t_{table.name}[i] = m.{table.name}[i];\n"
        "      }\n"
        for table in face.tables
    )
    return mx.fast.metal_kernel(
        name=f"omlx_format_staged_tables_{face.key}",
        input_names=inputs,
        output_names=[f"t_{table.name}" for table in face.tables],
        header=matrices.header,
        source=matrices.declarations
        + "    if (thread_index_in_threadgroup == 0) {\n"
        + copies
        + "    }\n",
    )


def staged_tables(face: Dot | Tile, operands, dtype: mx.Dtype) -> dict[str, mx.array]:
    """Each of ``face``'s tables as a kernel stages it, read back through the
    face's member of the table's name: one array of the table's dtype and
    length, by name. Its bits are the table's ``bits`` when the face
    declares its members as the page says.

    ``operands`` are a matrix's weight and the face's tensors, as for
    ``dot_rows`` (the face is declared over them), and ``dtype`` the
    activations' type ``T`` the face is compiled for."""
    if not isinstance(face, (Dot, Tile)):
        raise ValueError(f"{face!r} is not a Dot or a Tile")
    if dtype not in formats.FLOATS:
        raise ValueError(f"dtype={dtype} is not a float type of the GPU")
    if not face.tables:
        raise ValueError(f"face {face.key} has no tables")
    operands, names = tuple(operands), ("weight", *face.tensors)
    if len(operands) != len(names):
        raise ValueError(
            f"face {face.key} reads {', '.join(names)}; operands given: {len(operands)}"
        )
    for name, operand in zip(names, operands, strict=True):
        if operand.size < 8:
            raise ValueError(
                f"face {face.key}: {name} has {operand.size} elements; MLX binds an "
                "input of fewer than 8 as constants, not as a device buffer"
            )
    tables = _staged_tables_kernel(face)(
        inputs=list(operands),
        template=[("T", dtype)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(len(table.values),) for table in face.tables],
        output_dtypes=[table.dtype for table in face.tables],
    )
    return {table.name: got for table, got in zip(face.tables, tables, strict=True)}


def _checked(face: Dot, operands, x: mx.array) -> tuple:
    """``operands`` as a tuple, or a ``ValueError``: one vector ``x`` of a
    dtype the face is exact for, the face's operands, each of 8 elements or
    more."""
    if x.ndim != 1:
        raise ValueError(f"x is one vector of K activations, not of shape {x.shape}")
    operands, names = tuple(operands), ("weight", *face.tensors)
    if len(operands) != len(names):
        raise ValueError(
            f"face {face.key} reads {', '.join(names)}; operands given: {len(operands)}"
        )
    if not face.exact_for(x.dtype):
        exact = ", ".join(str(dtype) for dtype in face.dtypes or formats.FLOATS)
        raise ValueError(f"face {face.key} is exact for {exact}, x is {x.dtype}")
    for name, operand in zip(("x", *names), (x, *operands), strict=True):
        if operand.size < 8:
            raise ValueError(
                f"face {face.key}: {name} has {operand.size} elements; MLX binds an "
                "input of fewer than 8 as constants, not as a device buffer"
            )
    return operands


def dot_rows(
    face: Dot,
    operands,
    x: mx.array,
    n: int,
    packs: tuple[int, int] = (1, 1),
    staged: bool = False,
) -> mx.array:
    """The products of ``x`` (K activations) with the first ``n`` rows of a
    matrix, through the row skeleton.

    ``operands`` are the matrix's weight and the face's tensors, in order.
    A launch takes ``packs[0]`` rows as the skeleton's first pack and the
    next ``packs[1]`` as its second, one row or more in all; ``n`` is one
    or more times their sum, and no more than the matrix's rows (which is
    not checked). ``x`` is one vector of a dtype the face is exact for; it
    and each operand have 8 elements or more. With ``staged``, the kernel
    copies ``x`` into threadgroup memory and the face loads it from there,
    as a kernel that stages its inputs does: the rows are the same, bit for
    bit; ``x`` then fits in 32 KiB. Anything else is refused with a
    ``ValueError``. The kernel is compiled for ``face`` and K.
    """
    na, nb = packs
    if min(na, nb) < 0 or na + nb == 0:
        raise ValueError(f"packs={packs}: two counts of rows, one row or more in all")
    if n <= 0 or n % (na + nb):
        raise ValueError(f"n={n} is not one launch or more of {na + nb} rows")
    operands = _checked(face, operands, x)
    if staged and x.nbytes > _THREADGROUP_BYTES:
        raise ValueError(
            f"x of {x.nbytes} bytes does not fit in the {_THREADGROUP_BYTES} bytes "
            "of a threadgroup's memory"
        )
    return _dot_rows_kernel(face, staged)(
        inputs=[x, *operands],
        template=[("T", x.dtype), ("K", x.shape[-1]), ("NA", na), ("NB", nb)],
        grid=(32, n // (na + nb), 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[x.dtype],
    )[0]


@cache
def _shared_rows_kernel(face: Dot, unrolled: bool):
    matrices = kernels.Matrices(unrolled=unrolled)
    return mx.fast.metal_kernel(
        name=f"omlx_format_shared_rows_{face.key}",
        input_names=["x", *matrices.add("m", face, shared=True)],
        output_names=["y"],
        header=matrices.header,
        source=matrices.declarations + _SHARED_ROWS,
    )


def shared_rows(
    face: Dot,
    operands,
    x: mx.array,
    n: int,
    rps: int = 1,
    unrolled: bool = False,
) -> mx.array:
    """The products of each of ``x``'s rows (one to 16 vectors of K
    activations) with the first ``n`` rows of a matrix, through the
    shared-rows skeleton (``kernels.qmv_shared``): each block's weights
    decoded once for all the vectors, then applied to each. ``rps`` output
    rows a simdgroup (the last pack holds the rest when ``n`` is not whole
    packs); ``unrolled``: the skeleton's loops are unrolled in full. A face that
    keeps the rule gives each vector's row ``dot_rows``' bits. ``n`` is one
    or more and no more than the matrix's rows (not checked); the rest is
    refused as ``dot_rows`` refuses it."""
    if x.ndim != 2 or not 1 <= x.shape[0] <= 16:
        raise ValueError(f"x is 1 to 16 vectors of K activations, not {x.shape}")
    operands = _checked(face, operands, x[0])
    if n <= 0 or rps <= 0:
        raise ValueError(f"n={n}, rps={rps}: one row or more of each")
    rows, k = x.shape
    return _shared_rows_kernel(face, unrolled)(
        inputs=[x, *operands],
        template=[
            ("T", x.dtype),
            ("K", k),
            ("ROWS", rows),
            ("RPS", rps),
            ("VPT", face.vpt),
            ("PARTIAL", n % rps != 0),
            ("N", n),
        ],
        grid=(32, -(-n // rps), 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )[0]


# Each lane's part of rows: madd(0, row(0)) over the VPT inputs of lane
# index j = 32 * b + l. STEPPED: lane l's cursor, stepped b blocks, as the
# row skeleton walks it. Else: a cursor built at j, in thread j, as a kernel
# that splits K or walks a row in one thread builds it.
_LANE_PARTS_HEADER = r"""
template <typename T, int K, bool STEPPED, typename D>
METAL_FUNC void omlx_lane_parts(
    D m, size_t row, uint at, const device T* x, device float* parts) {
  using L = typename D::template Lanes<K>;
  constexpr int J = K / L::VPT;
  float xt[L::VPT];
  parts += row * K;
  if constexpr (STEPPED) {
    L a(m, row, at);
    for (int j = int(at); j < J; j += 32) {
      const auto aux = L::load(x + j * L::VPT, xt);
      parts[j] = L::madd(0.0f, a.row(0), xt, aux);
      a.step();
    }
  } else if (int(at) < J) {
    const L c(m, row, at);
    const auto aux = L::load(x + int(at) * L::VPT, xt);
    parts[at] = L::madd(0.0f, c.row(0), xt, aux);
  }
}
"""


@cache
def _lane_parts_kernel(face: Dot, stepped: bool):
    matrices = kernels.Matrices()
    inputs = ["x", *matrices.add("m", face)]
    return mx.fast.metal_kernel(
        name=f"omlx_format_lane_parts_{face.key}",
        input_names=inputs,
        output_names=["parts"],
        header=matrices.header + _LANE_PARTS_HEADER,
        source=matrices.declarations
        + "omlx_lane_parts<T, K, STEPPED>(m, size_t(thread_position_in_grid.y), "
        "thread_position_in_grid.x, x, parts);\n",
    )


def lane_parts(face: Dot, operands, x: mx.array, n: int) -> tuple[mx.array, mx.array]:
    """Each lane's part of the first ``n`` rows of a matrix, read twice:
    ``madd(0, row(0), xt, aux)`` over the ``VPT`` inputs of each lane index
    ``j = 32 * b + l``, by lane ``l``'s cursor stepped ``b`` blocks, as the
    row skeleton walks it, and by a cursor built at ``j``, in a thread of
    its own. A face that keeps the cursor rule gives the two the same bits.

    Returns two float32 arrays of shape ``(n, K)``: column ``j`` holds lane
    index ``j``'s part for ``j`` below ``K / VPT``, zeros after.
    ``operands`` and ``x`` are as ``dot_rows`` takes them; ``n`` is one or
    more and no more than the matrix's rows (not checked)."""
    operands = _checked(face, operands, x)
    if n <= 0:
        raise ValueError(f"n={n}: one row or more")
    k = x.shape[-1]
    return tuple(
        _lane_parts_kernel(face, stepped)(
            inputs=[x, *operands],
            template=[("T", x.dtype), ("K", k), ("STEPPED", stepped)],
            grid=(32 if stepped else k, n, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(n, k)],
            output_dtypes=[mx.float32],
            init_value=0,
        )[0]
        for stepped in (True, False)
    )


# finish(0, sums[i]) of row `row`, in each lane. finish reads nothing that
# step() moves, so the cursor is where the lane starts.
_FINISH_ROWS_HEADER = r"""
template <typename T, int K, int S, typename D>
METAL_FUNC void omlx_finish_rows(
    D m, size_t row, uint lane, const device float* sums, device float* out) {
  using L = typename D::template Lanes<K>;
  const L a(m, row, lane);
  out += (row * 32 + lane) * S;
  for (int i = 0; i < S; i++) {
    out[i] = a.finish(0, sums[i]);
  }
}
"""


@cache
def _finish_rows_kernel(face: Dot):
    matrices = kernels.Matrices()
    inputs = ["sums", *matrices.add("m", face)]
    return mx.fast.metal_kernel(
        name=f"omlx_format_finish_rows_{face.key}",
        input_names=inputs,
        output_names=["out"],
        header=matrices.header + _FINISH_ROWS_HEADER,
        source=matrices.declarations
        + "omlx_finish_rows<T, K, S>(m, size_t(thread_position_in_grid.y), "
        "thread_position_in_grid.x, sums, out);\n",
    )


def finish_rows(
    face: Dot, operands, k: int, n: int, sums: mx.array, dtype: mx.Dtype
) -> mx.array:
    """``finish(0, s)`` of each of the first ``n`` rows of a matrix of K
    inputs, for each sum ``s`` in ``sums`` (8 or more float32 values), in
    each of a simdgroup's 32 lanes: a float32 array of shape ``(n, 32,
    len(sums))``. ``finish`` is
    a scale: ``finish(r, s)`` is ``s`` times a value of the row alone, the
    same in every lane. The kernel is compiled for ``face``, ``dtype`` (a
    dtype the face is exact for), K and the count of sums; ``operands`` are
    as ``dot_rows`` takes them."""
    if sums.ndim != 1 or sums.dtype != mx.float32 or sums.size < 8:
        raise ValueError(
            f"sums are 8 or more float32 values, not {sums.dtype} {sums.shape}"
        )
    operands = _checked(face, operands, mx.zeros((max(k, 8),), dtype))
    if k <= 0 or n <= 0:
        raise ValueError(f"k={k}, n={n}: one input and one row or more")
    return _finish_rows_kernel(face)(
        inputs=[sums, *operands],
        template=[("T", dtype), ("K", k), ("S", sums.size)],
        grid=(32, n, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(n, 32, sums.size)],
        output_dtypes=[mx.float32],
    )[0]


def rows_stack(
    fmt, first: dict, second: dict, k: int, bits: int, group_size: int, x=None
) -> None:
    """Checks the rows rule's promise for two ``(n, k)`` matrices of ``fmt``
    (their ``weight`` and the format's tensors, by name, as ``layout`` gives
    them): stacked along N, their decode is the two decodes one after the
    other, bit for bit, and with ``x`` (K activations of a dtype the face is
    exact for) so are their dot face's rows. With ``x`` the helper builds
    format layers, so call it inside ``testing.verified(...)`` in a format
    test. With ``x=None`` it only calls the format's decode after checking
    both matrices against the declared tensor names, shapes and dtypes.
    Raises an ``AssertionError`` that names what differs, or that the
    format is not ``stackable``."""
    weight = first.get("weight")
    if not isinstance(weight, mx.array) or not weight.ndim:
        raise AssertionError(f"quant format {fmt.mode}: first matrix needs weight rows")
    n = weight.shape[0]
    layout = formats._stackable_layout(fmt, k, n, bits, group_size)
    if layout is None:
        raise AssertionError(f"quant format {fmt.mode}: not stackable at n={n}, k={k}")
    expected = {name: (tuple(shape), dtype) for name, (shape, dtype) in layout.items()}
    for side, matrix in (("first", first), ("second", second)):
        actual = {
            name: (t.shape, t.dtype) if isinstance(t, mx.array) else type(t).__name__
            for name, t in matrix.items()
        }
        if actual != expected:
            raise AssertionError(
                f"quant format {fmt.mode}: {side} matrix has {actual}, "
                f"expected {expected}"
            )
    stacked = {
        name: mx.concatenate([first[name], second[name]], axis=0) for name in first
    }
    whole = fmt.decode(stacked, bits, group_size)
    parts = [fmt.decode(m, bits, group_size) for m in (first, second)]
    if not _same_bits(whole, mx.concatenate(parts)):
        raise AssertionError(
            f"quant format {fmt.mode}: the decode of two stacked matrices is not "
            "their two decodes"
        )
    if x is None:
        return
    rows = []
    for tensors in (stacked, first, second):
        layer = FormatLinear(fmt, bits, group_size, k, tensors)
        face = fmt.dot(layer, k, layer.output_dims)
        if face is None:
            raise AssertionError(f"quant format {fmt.mode}: no dot face to compare")
        operands = [layer[name] for name in ("weight", *face.tensors)]
        rows.append(dot_rows(face, operands, x, layer.output_dims, (1, 0)))
    if not _same_bits(rows[0], mx.concatenate(rows[1:])):
        raise AssertionError(
            f"quant format {fmt.mode}: the dot rows of two stacked matrices are not "
            "their two matrices' rows"
        )


def _same_bits(a: mx.array, b: mx.array) -> bool:
    """Equal shapes, dtypes and bits (a NaN equals a NaN of its bits)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if mx.issubdtype(a.dtype, mx.floating):
        bits = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
        a, b = a.view(bits), b.view(bits)
    return mx.array_equal(a, b).item()


# Each weight of the rows, one a thread: weight w of row rows[r], of the
# kernel's K, the grid's width.
_TILE_ROWS = r"""
    const uint w = thread_position_in_grid.x;
    const uint r = thread_position_in_grid.y;
    const size_t at = size_t(r) * size_t(K) + w;
    omlx_tile_rows<T, K>(m, size_t(rows[r]), int(w), w32 + at, wt + at);
"""

# Weight w of row `row`, as a tile kernel reads it: the code's run, its
# group's parameters, then the weight, stored as a float32 and rounded once
# to T. Its position in the run is not known when the kernel is compiled.
# As a kernel does, it calls params and fetch more than once on one cursor,
# into arrays of P and Codes, another group and run first (where the row has
# more than one), and keeps the second answers.
_TILE_ROWS_HEADER = r"""
template <typename T, int K, typename F>
METAL_FUNC void omlx_tile_rows(
    const F m, size_t row, int w, device float* w32, device T* wt) {
  if constexpr (omlx_tile_face<F, K>::value) {
    using R = typename F::template Runs<K>;
    constexpr int GROUPS = (K - 1) / R::GROUP + 1;
    const R runs(m, row);
    typename R::P p[2];
    typename R::Codes c[2];
    p[0] = runs.params((w / R::GROUP + 1) % GROUPS);
    c[0] = runs.fetch((w / R::RUN + 1) % (K / R::RUN));
    p[1] = runs.params(w / R::GROUP);
    c[1] = runs.fetch(w / R::RUN);
    thread const auto& params = p[1];
    thread const auto& codes = c[1];
    const float v = R::weight(params, codes, w % R::RUN);
    *w32 = v;
    *wt = static_cast<T>(v);
  }
}
"""


@cache
def _tile_rows_kernel(face: Tile):
    matrices = kernels.Matrices()
    inputs = ["rows", *matrices.add("m", face)]
    return mx.fast.metal_kernel(
        name=f"omlx_format_tile_rows_{face.key}",
        input_names=inputs,
        output_names=["w32", "wt"],
        header=matrices.header + _TILE_ROWS_HEADER,
        source=matrices.declarations + _TILE_ROWS,
    )


def tile_rows(
    face: Tile, operands, k: int, rows, dtype: mx.Dtype
) -> tuple[mx.array, mx.array]:
    """Every weight of some rows of a matrix of K inputs, read through a tile
    face as a tile kernel reads it: ``weight(params(g), fetch(run), i)``.

    ``operands`` are the matrix's weight and the face's tensors, in order.
    ``rows`` are the rows' indices (in a stack, the row ``n`` of matrix
    ``e`` is ``e * N + n``). Each index is an int in [0, 2**63) and names
    a row of the matrix (the matrix bound is unchecked).
    K is an int, a whole number of runs of 8 below 2^31. Returns the weights
    twice, each array of shape ``(len(rows), k)``: as ``weight`` gives them,
    in float32, and rounded once to ``dtype`` (float32, float16 or bfloat16,
    and one the face is exact for), as a kernel
    stores them for activations of that dtype. Each operand has 8 elements
    or more. Anything else is refused with a ``ValueError``. The kernel is
    compiled for ``face``, ``dtype`` and K; the rows are values it is
    given.
    """
    if not isinstance(face, Tile):
        raise ValueError(f"{face!r} is not a Tile")
    if not (isinstance(k, int) and 0 < k < 1 << 31 and k % 8 == 0):
        raise ValueError(f"k={k} is not one run or more of 8 inputs")
    try:
        rows, operands = list(rows), tuple(operands)
    except TypeError as error:
        raise ValueError(f"rows and operands are sequences: {error}") from None
    if not rows or not all(
        isinstance(r, int) and not isinstance(r, bool) and 0 <= r < 1 << 63
        for r in rows
    ):
        raise ValueError(f"rows={rows} are not one row index or more")
    names = ("weight", *face.tensors)
    if len(operands) != len(names):
        raise ValueError(
            f"face {face.key} reads {', '.join(names)}; operands given: {len(operands)}"
        )
    if dtype not in (mx.float32, mx.float16, mx.bfloat16):
        raise ValueError(f"dtype={dtype} is not a float type of the GPU")
    if not face.exact_for(dtype):
        exact = ", ".join(str(exact) for exact in face.dtypes)
        raise ValueError(f"face {face.key} is exact for {exact}, not {dtype}")
    for name, operand in zip(names, operands, strict=True):
        if not isinstance(operand, mx.array):
            raise ValueError(f"face {face.key}: {name} is not an mx.array")
        if operand.size < 8:
            raise ValueError(
                f"face {face.key}: {name} has {operand.size} elements; MLX binds an "
                "input of fewer than 8 as constants, not as a device buffer"
            )
    return tuple(
        _tile_rows_kernel(face)(
            inputs=[mx.array(rows, dtype=mx.uint64), *operands],
            template=[("T", dtype), ("K", k)],
            grid=(k, len(rows), 1),
            threadgroup=(min(k, 256), 1, 1),
            output_shapes=[(len(rows), k), (len(rows), k)],
            output_dtypes=[mx.float32, dtype],
        )
    )


@dataclass(frozen=True, eq=False)
class LayerCall:
    """Rows of a format layer, as ``layer_calls`` records them where they
    ran: the layer, the face whose launch ran them (or the replay of the
    launch the layer kept), and what that launch's Metal is instantiated
    with beside the face (``metal``: x's type; for a stack, the tuple of
    x's type, the pairs of a row of x and whether there are fewer than 8
    pairs). ``face`` and ``metal`` are None for rows of the format's decode.
    Two are equal when they are of the same layer, with equal faces and
    Metal; like the layer, a dict of its arrays, a record has no hash."""

    layer: Any
    face: Dot | None = None
    metal: Any = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, LayerCall):
            return NotImplemented
        return self.layer is other.layer and (self.face, self.metal) == (
            other.face,
            other.metal,
        )

    def __repr__(self) -> str:
        layer = self.layer
        rows = (
            f"{type(layer).__name__} {layer.mode} "
            f"{layer.input_dims} -> {layer.output_dims}"
        )
        if self.face is None and self.metal is None:
            return f"LayerCall({rows}: decode)"
        key = self.face.key if self.face is not None else None
        return f"LayerCall({rows}: face {key}, Metal {self.metal})"


class _Frame:
    """A format layer's call running in a recording thread: the layer, and
    what the call has recorded so far, the thread's blocks' when it
    returns."""

    __slots__ = ("layer", "calls")

    def __init__(self, layer) -> None:
        self.layer, self.calls = layer, []


class _Thread:
    """What a thread records in: its open blocks of ``layer_calls``,
    innermost last (a tuple, replaced under ``_WRAPPING`` when a block
    starts or ends, which may be in another thread), and the calls of
    format layers running in it, innermost last."""

    __slots__ = ("blocks", "frames")

    def __init__(self) -> None:
        self.blocks: tuple[list[LayerCall], ...] = ()
        self.frames: list[_Frame] = []


class _Recording(threading.local):
    def __init__(self) -> None:
        self.thread = _Thread()


_RECORDING = _Recording()
# Held to wrap the layers when a first block opens, in any thread, to give
# them back their own when the last one ends, and to edit a thread's blocks.
_WRAPPING = threading.Lock()
_wrapped: list = []  # (owner, name, its own), in the order they were wrapped
_open = 0  # blocks open in every thread


def _record(thread: _Thread, calls: list[LayerCall]) -> None:
    for block in thread.blocks:
        block.extend(calls)


def _described(linear, plan, arrays) -> Any:
    """What a launch's Metal is built for, by the helpers the launch itself
    describes it with: ``arrays`` are x, then a stack's indices."""
    x, *indices = arrays
    return linear._metal(x, linear._pairs(plan.k, x, *indices) if indices else None)


class _Replays(Mapping):
    """A proof's kept graphs as a recording thread's running call reads
    them: the thread's own, not a copy, in which a graph that the call
    finds (``get``) records its rows in the call when it is called."""

    def __init__(self, kept: dict, linear, frame: _Frame, plan) -> None:
        self._kept, self._linear, self._frame, self._plan = kept, linear, frame, plan

    def __getitem__(self, signature):
        return self._kept[signature]

    def __iter__(self):
        return iter(self._kept)

    def __len__(self) -> int:
        return len(self._kept)

    def get(self, signature, default=None):
        graph = self._kept.get(signature)
        if graph is None:
            return default
        linear, frame, plan = self._linear, self._frame, self._plan

        def replayed(*arrays):
            y = graph(*arrays)
            launched = arrays[: len(arrays) - len(plan.operands)]
            described = _described(linear, plan, launched)
            frame.calls.append(LayerCall(frame.layer, plan.face, described))
            return y

        return replayed


def _framed(call, layer, *args):
    """``call(layer, *args)``, a format layer's own call, in a frame of the
    calling thread while it has a block open: what the call records is
    recorded when it returns (in the call that made it, if any), and
    dropped when it raises."""
    thread = _RECORDING.thread
    if not thread.blocks:
        return call(layer, *args)
    frames, frame = thread.frames, _Frame(layer)
    try:
        frames.append(frame)
        y = call(layer, *args)
    finally:
        # Gone if the thread's last block ended meanwhile.
        if frame in frames:
            del frames[frames.index(frame) :]
    if frames:
        frames[-1].calls += frame.calls
    else:
        _record(thread, frame.calls)
    return y


def _wrappers(linear) -> list:
    """What a block wraps, and its wrapper: the layers' two calls (a frame
    for each), a kept graph's hit, a launch that returns rows and the
    decode, each wrapper recording in the calling thread's blocks alone,
    with the signature of what it wraps."""
    linear_call = vars(linear.FormatLinear)["__call__"]
    switch_call = vars(linear.FormatSwitchLinear)["__call__"]
    graphs = vars(kernels.Proof)["graphs"]
    kept = graphs.fget
    launch = vars(linear)["_launch"]
    decoded = vars(linear._FormatLayer)["_decoded"]

    @wraps(linear_call)
    def linear_called(self, x):
        return _framed(linear_call, self, x)

    @wraps(switch_call)
    def switch_called(self, x, indices, sorted_indices=False):
        return _framed(switch_call, self, x, indices, sorted_indices)

    def replays(proof):
        graphs, frames = kept(proof), _RECORDING.thread.frames
        if not frames:
            return graphs
        # The running call's plan, as its layer keeps it.
        plan = frames[-1].layer.__dict__.get(linear._KEPT, (None,))[0]
        if getattr(plan, "proof", None) is proof:
            return _Replays(graphs, linear, frames[-1], plan)
        return graphs

    @wraps(launch)
    def _launch(plan, signature, describe, stand_in, *arrays):
        # The running call, before the launch runs any other.
        frames = _RECORDING.thread.frames
        frame = frames[-1] if frames else None
        y = launch(plan, signature, describe, stand_in, *arrays)
        if y is not None and frame is not None:
            described = _described(linear, plan, arrays)
            frame.calls.append(LayerCall(frame.layer, plan.face, described))
        return y

    @wraps(decoded)
    def _decoded(self, indices=None):
        weights = decoded(self, indices)
        thread = _RECORDING.thread
        if thread.frames:
            thread.frames[-1].calls.append(LayerCall(self))
        else:
            _record(thread, [LayerCall(self)])
        return weights

    return [
        (linear.FormatLinear, "__call__", linear_called),
        (linear.FormatSwitchLinear, "__call__", switch_called),
        (kernels.Proof, "graphs", property(replays, doc=graphs.__doc__)),
        (linear, "_launch", _launch),
        (linear._FormatLayer, "_decoded", _decoded),
    ]


def _wrap() -> None:
    """Every wrapper, or none: an exception on the way gives back what was
    wrapped. What an end interrupted while it gave the layers back left
    wrapped is given back first."""
    from . import linear

    _unwrap()
    wrappers = _wrappers(linear)
    try:
        for owner, name, wrapper in wrappers:
            _wrapped.append((owner, name, vars(owner)[name]))
            setattr(owner, name, wrapper)
    except BaseException:
        _unwrap()
        raise


def _unwrap() -> None:
    while _wrapped:
        owner, name, own = _wrapped[-1]
        setattr(owner, name, own)
        _wrapped.pop()


@contextmanager
def layer_calls():
    """Record where the rows of format layers ran in this thread, in the
    block, in order, each a ``LayerCall``: a launch of the face, or the
    replay of the launch a layer kept (``face``), or the format's decode
    (``face`` None: more rows than a launch takes, no face, a face not
    exact for x's type, one that did not compile). Each is recorded where
    it ran, tagged with its layer, when the layer's call returns. The
    block's value is the list so far.

    The layers are wrapped while a block is open, in any thread; a thread
    without a block runs as it does without, and when the last block ends
    the layers have their own back: outside a block a call costs nothing
    more. A call that raises records nothing, nor do the calls it made.
    Nor is what runs no Python of the layer recorded: a call that a
    function compiled by ``mx.compile`` replays (it is recorded once, as it
    is traced), or a fused kernel's launch over a layer's tensors."""
    global _open
    calls: list[LayerCall] = []
    # This thread's: the block's end, in whichever thread, ends it here.
    thread, opened = _RECORDING.thread, False
    try:
        with _WRAPPING:
            if not _open:
                _wrap()
            _open += 1
            opened = True
            thread.blocks = (*thread.blocks, calls)
        yield calls
    finally:
        with _WRAPPING:
            blocks = thread.blocks
            thread.blocks = tuple(block for block in blocks if block is not calls)
            if not thread.blocks:
                thread.frames.clear()
            if opened:
                _open -= 1
                if not _open:
                    _unwrap()


class _Swapped:
    """A registry of the process (``module.name``) that blocks replace with
    one of their own while they are open: in any thread, and ending in any
    order. It is the latest of the blocks still open, or, once they have all
    ended, the one that the first of them replaced. Each block is its own
    entry, so two blocks of one registry end apart; an exception or an
    interrupt as a block starts leaves the registry as the blocks still
    open have it."""

    def __init__(self, module, name: str, lock=None) -> None:
        self.module, self.name = module, name
        # Held to swap the registry, and by whoever reads and writes it as
        # one step.
        self.lock = threading.Lock() if lock is None else lock
        self.open: list = []  # the open blocks, oldest first, each (registry,)

    @contextmanager
    def block(self, registry):
        entry = None
        try:
            with self.lock:
                if not self.open:
                    self.replaced = getattr(self.module, self.name)
                entry = (registry,)
                self.open.append(entry)
                setattr(self.module, self.name, registry)
            yield registry
        finally:
            if entry is not None:
                with self.lock:
                    self.open = [other for other in self.open if other is not entry]
                    latest = self.open[-1][0] if self.open else self.replaced
                    setattr(self.module, self.name, latest)


_FRESH_PROOFS = _Swapped(kernels, "_PROOFS")
# Swapped under the lock that omlx.formats.dot notes a defect under: a note
# lands in the set it read, the one of the latest block then open.
_DEFECTS = _Swapped(formats, "_DOT_DEFECTS", formats._NOTING)


@contextmanager
def fresh_proofs():
    """Give the block a registry of proofs of its own (``kernels.proof``),
    and the registry it replaced back when the block ends.

    A launch that fails keeps every layer and kernel of its key on the
    composed path for the life of the process: a test that fails a face
    would leave the tests after it on the decode. oMLX runs each test under
    ``tests/`` in this block, and so can a plugin, from an autouse fixture.
    The block's value is its registry.

    The registry is the process's: while the block is open, every thread's
    kernels prove their launches in it. Blocks may end in any order, in any
    thread: the registry is then the latest of those still open, or the one
    that the first replaced. A layer keeps the plan of its first call, proof
    included: a layer called before the block keeps that call's proof."""
    with _FRESH_PROOFS.block({}) as proofs:
        yield proofs
