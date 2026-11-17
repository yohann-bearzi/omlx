# SPDX-License-Identifier: Apache-2.0
"""Checks for a format's tests. ``dot_rows`` runs a face in the skeletons
oMLX's kernels are built from, outside any model, ``shared_rows`` runs
several vectors through ``decode`` and ``apply``, and ``tile_rows`` reads
every weight of rows through a tile face. ``staged_tables`` reads a face's
tables back as a kernel stages them. ``no_dot_defects`` fails a test in
which a format's ``dot()`` or ``tile()`` was defective. ``layer_calls`` records
where format layers' rows ran: their face's launch, or the format's decode.
``fresh_proofs`` gives a test proofs of its own. ``verified`` has oQ take model families
as verified with quant formats for a test's checkpoints.

Nothing here is imported by the layers or the kernels: what a check needs of
them it wraps for the length of a block.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache, wraps
from typing import Any

import mlx.core as mx

from .. import formats
from . import Dot, Tile, kernels

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


@contextmanager
def verified(*families: str):
    """For the block, oQ writes checkpoints of ``families`` (model types, in
    any spelling) in a quant format as it does those verified with them, and
    as it then did after. A test's checkpoint of a family oMLX patches (a
    tiny one, to test a format with) is written so. As a decorator, it holds
    for one test; in a fixture, ``with verified(...): yield``."""
    from .. import oq

    kept = oq._QUANT_FORMAT_VERIFIED
    oq._QUANT_FORMAT_VERIFIED = kept | {oq._family(family) for family in families}
    try:
        yield
    finally:
        oq._QUANT_FORMAT_VERIFIED = kept


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
    F m, size_t row, int w, device float* w32, device T* wt) {
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
    const float v = R::weight(p[1], c[1], w % R::RUN);
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
    ``e`` is ``e * N + n``), none past the matrix's (which is not checked).
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
    rows = list(rows)
    if not rows or not all(
        isinstance(r, int) and not isinstance(r, bool) and r >= 0 for r in rows
    ):
        raise ValueError(f"rows={rows} are not one row index or more")
    operands, names = tuple(operands), ("weight", *face.tensors)
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
