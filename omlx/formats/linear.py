# SPDX-License-Identifier: Apache-2.0
"""Linear layers whose weights are kept in a registered format: what
``nn.QuantizedLinear`` and mlx-vlm's ``QuantizedSwitchLinear`` are to the
formats MLX knows.

A layer holds a matrix's tensors under the names its format gives them, and
asks the registry how to run them:

- the rows of a decode step, or of the short blocks around one, go through
  the format's dot face in the mat-vec skeleton, one launch for all of them,
  each row with the bits it has alone;
- everything else (more rows of one matrix, a layout the face does not
  cover, activations of a type it is not exact for, a format without the
  face, a face that did not compile) is the format's ``decode``, cast to the
  activations' type, then MLX's matmul (its ``gather_mm`` from as many
  indices as a switch layer's stack has matrices).
"""

from __future__ import annotations

import logging
import math
from functools import cache
from typing import Any, NamedTuple

import mlx.core as mx
import mlx.nn as nn

from .. import formats
from . import Dot, kernels

logger = logging.getLogger(__name__)

# The rows of one ``FormatLinear`` call that go through the dot face: a decode
# step and the short blocks around it (an MTP verify window). Each has the
# bits of its one-row call, whatever rows it is launched with.
MAX_ROWS = 16

# One threadgroup per pair (a row of x and the matrix it meets) and tile of 8
# of the pair's outputs: 4 for each of the threadgroup's two simdgroups.
# __MATRIX__ is the pair's matrix in the stack, __X_ROW__ its row of x.
_STACKED = r"""
    static_assert(
        PAIRS_PER_ROW >= 1,
        "format switch linear: PAIRS_PER_ROW is the matrices of a row of x, 1 or more");
"""
_SOURCE = r"""
    constexpr int RPS = 4;  // output rows of a simdgroup
    static_assert(
        N_SIZE % (2 * RPS) == 0,
        "format linear: N must be whole tiles of 8 outputs");
    const uint pair = threadgroup_position_in_grid.y;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const size_t out_row =
        size_t(threadgroup_position_in_grid.z) * (2 * RPS) + simd_gid * RPS;
    float result[RPS];
    qmv_rows<T, K_SIZE, RPS, 0>(
        m, size_t(__MATRIX__) * N_SIZE + out_row, m, 0,
        x + size_t(__X_ROW__) * K_SIZE, simd_lid, result);
    if (simd_lid == 0) {
      store_rows(omlx_iota<RPS>(), result, y + size_t(pair) * N_SIZE + out_row);
    }
"""


# Two to 16 rows of x in one launch, G of them a threadgroup: each block of
# the matrix decoded once for the G rows (the shared-rows skeleton), each
# row its one-row bits. RPS output rows a simdgroup, two simdgroups a tile
# of 8, as the one-row kernel; the skeleton is unrolled (G rows within
# kernels.ROW_VALUES_BUDGET, inside kernels.unrolled_tile_ok). Where N is not whole tiles (PARTIAL), the last
# tile's simdgroups hold the rows that are left, or none, as MLX's qmv does
# for its last tile.
_SHARED_SOURCE = r"""
    constexpr int RPS = 4;
    constexpr bool PARTIAL = N_SIZE % (2 * RPS) != 0;
    const size_t row0 = size_t(threadgroup_position_in_grid.y) * ROWS;
    const int out_row = int(threadgroup_position_in_grid.z) * (2 * RPS) +
        int(simdgroup_index_in_threadgroup) * RPS;
    if (PARTIAL && out_row >= N_SIZE) {
      return;
    }
    const int valid = PARTIAL ? min(RPS, N_SIZE - out_row) : RPS;
    float result[ROWS * RPS];
    qmv_shared<T, K_SIZE, ROWS, RPS, VPT, PARTIAL>(
        m, size_t(out_row), valid, x + row0 * K_SIZE, thread_index_in_simdgroup, result);
    if (thread_index_in_simdgroup == 0) {
      for (int i = 0; i < ROWS; i++) {
        device T* out = y + (row0 + i) * N_SIZE + out_row;
        if (PARTIAL) {
          for (int r = 0; r < valid; r++) {
            out[r] = static_cast<T>(result[i * RPS + r]);
          }
        } else {
          store_rows(omlx_iota<RPS>(), result + i * RPS, out);
        }
      }
    }
"""


@cache
def _shared_kernel(face: Dot):
    matrices = kernels.Matrices(unrolled=True)
    inputs = ["x", *matrices.add("m", face, shared=True)]
    return mx.fast.metal_kernel(
        name=f"omlx_format_linear_rows_{face.key}",
        input_names=inputs,
        output_names=["y"],
        header=matrices.header + kernels.STORE_ROWS,
        source=matrices.declarations + _SHARED_SOURCE,
    )


def _group(rows: int, vpt: int) -> int:
    """The rows of x a threadgroup of the shared launch takes: the most, up
    to 4, that divide ``rows`` and keep a lane's inputs of the group within
    kernels.ROW_VALUES_BUDGET (which keeps the unrolled tile in its
    envelope); 1 where none does (the one-row launch then runs each row).
    Measured (u7 spike, both Macs): at 16 values a lane, groups of 3 and 4
    ran 1.15-1.70 of the one-row launch where groups of 2 ran 0.76-0.98."""
    limit = kernels.ROW_VALUES_BUDGET // vpt
    return next((g for g in (4, 3, 2) if g <= limit and rows % g == 0), 1)


@cache
def _kernel(face: Dot, stacked: bool):
    matrices = kernels.Matrices()
    inputs = ["x", *matrices.add("m", face)]
    if stacked:
        inputs.append("rhs")
        matrix, x_row = "rhs[pair]", "pair / uint(PAIRS_PER_ROW)"
    else:
        matrix, x_row = "0", "pair"
    return mx.fast.metal_kernel(
        name=f"omlx_format_{'switch_linear' if stacked else 'linear'}_{face.key}",
        input_names=inputs,
        output_names=["y"],
        header=matrices.header + kernels.STORE_ROWS,
        source=matrices.declarations
        + (_STACKED if stacked else "")
        + _SOURCE.replace("__MATRIX__", matrix).replace("__X_ROW__", x_row),
    )


class _Plan(NamedTuple):
    face: Dot
    kernel: Any
    shared: Any  # 2 to 16 rows of a matrix in one launch; None for a stack
    operands: tuple  # the weight, then the face's tensors
    proof: kernels.Proof
    k: int
    n: int
    said: tuple[str, str]  # what a proof logs: the launches engaged, or failed


def _plan(layer) -> _Plan | None:
    """How ``layer``'s rows run through its format's dot face, or None: the
    format has no face for the matrix, or a launch could not take it."""
    k, n = layer.input_dims, layer.output_dims
    # MLX binds x as a buffer from 8 elements.
    if k < 8:
        return None
    face = formats.dot(layer, k, n)
    if face is None:
        return None
    operands = tuple(layer[name] for name in ("weight", *face.tensors))
    stacked = layer["weight"].ndim == 3
    # The one-row and the stack's launches are whole tiles of 8 outputs; a
    # matrix's N that is not runs in the shared-rows launch alone.
    if n % 8 and (stacked or face.vpt < 1):
        return None
    # What a launch is compiled from, beside what the layer keeps in ran.
    proof = kernels.proof(
        "format_switch_linear" if stacked else "format_linear",
        face,
        tuple(a.dtype for a in operands),
        k,
        n,
    )
    shared = None if stacked or face.vpt < 1 else _shared_kernel(face)
    kernel = _kernel(face, stacked)
    return _Plan(face, kernel, shared, operands, proof, k, n, _said(face.key))


@cache
def _said(key: str) -> tuple[str, str]:
    """What a proof of face ``key`` logs: the launches engaged, or failed."""
    return (
        f"quant format face {key}: rows kernel engaged",
        f"quant format face {key}: rows kernel failed; decode fallback",
    )


def _rows(
    kernel, k: int, n: int, x, operands, pairs, shape, rhs=None, per_row=1, tiles=None
):
    """One mat-vec per pair, ``pairs`` of them, as ``shape`` (which ends
    with N). Pair ``p`` is row ``p // per_row`` of ``x``, rows of K in any
    shape, with the matrix (``operands``), or with matrix ``rhs[p]`` of the
    stack. No reshape is put in the graph for the kernel's sake. A pair's
    outputs are ``tiles`` tiles of 8 from the first, or all of them: the
    grid is no part of the Metal, so a stand-in computes one."""
    dtype = x.dtype
    inputs = [x, *operands]
    template = [("T", dtype), ("K_SIZE", k), ("N_SIZE", n)]
    if rhs is not None:
        inputs.append(rhs)
        template.append(("PAIRS_PER_ROW", per_row))
    return kernel(
        inputs=inputs,
        template=template,
        grid=(32, 2 * pairs, n // 8 if tiles is None else tiles),
        threadgroup=(32, 2, 1),
        output_shapes=[shape],
        output_dtypes=[dtype],
    )[0]


class _Pairs(NamedTuple):
    """How a stack's launch pairs the rows of x with the indices."""

    pairs: int
    per_row: int  # the matrices of a row of x: its indices, or 1 if broadcast
    batch: tuple  # the leading axes of the result, ahead of x's rows (M) and N
    as_they_are: bool  # each row of x meets its own run of experts


def _pairs(k: int, x, indices) -> _Pairs | None:
    """How a stack's launch pairs ``x`` (rows of K in any shape) with
    ``indices``, from their shapes alone; None when there is no pair."""
    if not indices.size * x.size:
        return None
    shape, batch = x.shape, indices.shape
    lead, m = shape[:-2], shape[-2]
    own = len(lead)  # x's leading axes ahead of its trailing ones of 1
    while own and lead[own - 1] == 1:
        own -= 1
    # Each row of x meets its own run of experts: nothing is broadcast, in
    # memory or in shape. MLX binds indices of no axis as a scalar.
    as_they_are = m == 1 and len(batch) == len(lead) > 0 and batch[:own] == lead[:own]
    if as_they_are:
        pairs = indices.size
        return _Pairs(pairs, pairs // (x.size // k), batch, True)
    batch = mx.broadcast_shapes(lead, batch)
    return _Pairs(math.prod(batch) * m, 1, batch, False)


def _metal(x, pairs: _Pairs | None = None):
    """What a launch's Metal is instantiated with beside its proof's key:
    x's type, and for a stack the pairs of a row of x and whether there are
    fewer than 8 pairs (MLX binds fewer than 8 indices as constants: other
    Metal)."""
    return x.dtype if pairs is None else (x.dtype, pairs.per_row, pairs.pairs < 8)


def _linear(plan: _Plan, x):
    """A matrix's launch for ``x``: what its Metal is instantiated with
    beside the proof's key, and its graph, from ``x`` and the operands.
    Rows of x that a group of 2 to 4 divides run in the shared launch (x's
    type and the group), every other count in the one-row launch (x's
    type); a matrix whose N is not whole tiles of 8 runs every count in the
    shared launch (a group of 1 where none divides). The graph holds no
    array: it outlives the layers it is built for."""
    k, n = plan.k, plan.n
    pairs, shape = x.size // k, (*x.shape[:-1], n)
    group = 1 if plan.shared is None else _group(pairs, plan.face.vpt)
    if group > 1 or n % 8:
        shared, vpt = plan.shared, plan.face.vpt

        def graph(x, *operands):
            return _shared_rows(shared, k, n, vpt, x, operands, pairs, group, shape)

        return (x.dtype, group), graph
    kernel = plan.kernel

    def graph(x, *operands):
        return _rows(kernel, k, n, x, operands, pairs, shape)

    return _metal(x), graph


def _shared_rows(
    kernel, k: int, n: int, vpt: int, x, operands, rows, group, shape, tiles=None
):
    """``rows`` rows of ``x`` with the matrix, ``group`` of them a
    threadgroup, as ``shape``; ``tiles`` tiles of 8 outputs from the first,
    or all of them."""
    return kernel(
        inputs=[x, *operands],
        template=[
            ("T", x.dtype),
            ("K_SIZE", k),
            ("N_SIZE", n),
            ("ROWS", group),
            ("VPT", vpt),
        ],
        grid=(32, 2 * (rows // group), -(-n // 8) if tiles is None else tiles),
        threadgroup=(32, 2, 1),
        output_shapes=[shape],
        output_dtypes=[x.dtype],
    )[0]


def _linear_stand_in(plan: _Plan, metal):
    """The smallest launch of a matrix that MLX generates the Metal of
    ``metal`` for (x's type), over zeros: one vector, and the whole of each
    tensor. What follows the rows in a matrix's tensors is the format's
    own, so no part of a tensor is known to be all that a row reads."""
    operands = ((a.shape, a.dtype) for a in plan.operands)
    if isinstance(metal, tuple):
        # The shared launch: one group of rows.
        dtype, group = metal
        k, n, vpt = plan.k, plan.n, plan.face.vpt

        def launch(x, *tensors):
            # Its first tile of outputs.
            shape = (group, n)
            return _shared_rows(
                plan.shared, k, n, vpt, x, tensors, group, group, shape, 1
            )

        return kernels.zeros(((group, k), dtype), *operands), launch

    def launch(x, *tensors):
        # Its first tile of outputs.
        return _rows(plan.kernel, plan.k, plan.n, x, tensors, 1, (plan.n,), None, 1, 1)

    return kernels.zeros(((plan.k,), metal), *operands), launch


def _switch(plan: _Plan, x, indices):
    """A stack's launch for ``x`` and ``indices``, as ``_linear`` gives a
    matrix's, with the indices ahead of the operands; None when there is no
    pair to launch, or more than a grid takes. A negative index counts from
    the stack's end, as MLX's indexing does."""
    kernel, k, n = plan.kernel, plan.k, plan.n
    paired = _pairs(k, x, indices)
    # MLX's grid is int32, of 2 rows of threadgroups a pair.
    if paired is None or paired.pairs >= 2**30:
        return None
    pairs, per_row, batch, as_they_are = paired
    m = x.shape[-2]
    shape = (*batch, m, n)

    def graph(x, indices, *operands):
        if not as_they_are:
            x = mx.broadcast_to(x, (*batch, m, k))
            indices = mx.repeat(mx.broadcast_to(indices, batch), m)
        rhs = _wrapped(indices, operands[0].shape[0]).astype(mx.uint32)
        return _rows(kernel, k, n, x, operands, pairs, shape, rhs, per_row)

    return _metal(x, paired), graph


def _wrapped(indices, matrices: int):
    """``indices`` with each negative one counted from the end of a stack of
    ``matrices``, in a type wide enough to count them (a stack of 128 is no
    int8)."""
    if not mx.issubdtype(indices.dtype, mx.signedinteger):
        return indices
    if indices.dtype.size < 4:
        indices = indices.astype(mx.int32)
    return mx.where(indices < 0, indices + matrices, indices)


def _switch_stand_in(plan: _Plan, metal):
    """The smallest launch of a stack that MLX generates the Metal of
    ``metal`` for, over zeros. MLX binds an input of fewer than 8 elements
    as constants, which is other Metal: the stand-in has the fewest rows of
    x whose pairs are 8 or more, or fewer, as the call's are, and the
    fewest matrices that leave each tensor 8 elements. Every pair is with
    matrix 0."""
    dtype, per_row, few = metal
    rows = 1 if few else -(-8 // per_row)
    pairs = rows * per_row
    # Never more matrices than the stack has: formats.dot refuses a layer
    # with a tensor of fewer than 8 elements.
    matrices = max(-(-8 * a.shape[0] // a.size) for a in plan.operands)
    operands = (((matrices, *a.shape[1:]), a.dtype) for a in plan.operands)
    shape = (pairs, plan.n)

    def launch(x, *tensors):
        *tensors, rhs = tensors
        # Each pair's first tile of outputs.
        return _rows(
            plan.kernel, plan.k, plan.n, x, tensors, pairs, shape, rhs, per_row, 1
        )

    specs = (((rows, plan.k), dtype), *operands, ((pairs,), mx.uint32))
    return kernels.zeros(*specs), launch


def _launch(plan: _Plan | None, signature: tuple, describe, stand_in, *arrays):
    """``arrays`` (x, then a stack's indices) through the rows kernel, the
    first time for ``signature``, or None: the layer has no plan, its proof
    has failed, the face is not exact for x's type, there is nothing to
    launch, or the launch does not compile.

    The launch's graph (``describe``) is compiled by MLX and kept under
    ``signature``, which is what it depends on (the arrays' shapes, x's
    type and the indices' type), by the thread that builds it: a layer's
    call asks for the graph its thread kept first, and replays it.

    The first time its Metal is built, it is proven on ``stand_in``: the
    smallest launch that MLX generates the same Metal for, over zeros in
    the place of x, of the indices and of the layer's tensors. Nothing of
    the call is evaluated here: not ``arrays``, not a tensor, not the
    result. Any of them may belong to a trace (a model calls its layers in
    functions that MLX compiles, over its state too), where it cannot be
    evaluated, and what fails in one is raised where the caller evaluates
    the result."""
    if plan is None or plan.proof.failed or not plan.face.exact_for(arrays[0].dtype):
        return None
    launch = describe(plan, *arrays)
    if launch is None:
        return None
    metal, graph = launch
    proof = plan.proof
    if not proof.prove(metal, lambda: stand_in(plan, metal), logger, *plan.said):
        return None
    graph = mx.compile(graph)
    y = graph(*arrays, *plan.operands)
    proof.keep(signature, graph)
    return y


# Where a layer keeps what its calls work out of it once: its plan and its
# bias.
_KEPT = "_omlx_format_call"


class _FormatLayer(nn.Module):
    """What the two layers share: a format's tensors as frozen parameters,
    and the attributes of MLX's quantized layers.

    A layer works out its plan and finds its bias at its first call, and
    keeps them in its ``__dict__`` with the weight they were found with,
    until one of its arrays or attributes is assigned or removed
    (``__setitem__``, ``__delitem__``, ``pop``: what MLX's loads,
    ``update``, ``set_dtype`` and attribute writes go through; a dict's
    own mutators and ``del`` of an attribute too) or its
    weight is another array: a function that ``mx.compile`` traces over the
    layer's state (``inputs=``) swaps every array of it for a trace's, and
    back, behind those methods. A call then reads one dict entry and the
    weight. Threads: a layer that is being loaded is not called meanwhile;
    two threads that make a layer's first call at once each work out the
    same plan, and either keeps it."""

    _STACK = 0  # the axes ahead of one matrix's, in each of its tensors

    def __init__(
        self,
        mode: str,
        bits: int,
        group_size: int,
        input_dims: int,
        tensors: dict[str, mx.array],
        bias: mx.array | None = None,
    ) -> None:
        super().__init__()
        fmt = formats.get(mode)
        if fmt is None:
            raise ValueError(f"no quant format is registered for mode {mode!r}")
        names = ("weight", *fmt.tensors)
        if set(tensors) != set(names):
            raise ValueError(
                f"quant format {mode!r} keeps a matrix as {', '.join(names)}, "
                f"not as {', '.join(sorted(tensors))}"
            )
        weight = tensors["weight"]
        if weight.ndim != self._STACK + 2:
            raise ValueError(
                f"{type(self).__name__}: the weight has {self._STACK + 2} axes, "
                f"not the shape {weight.shape}"
            )
        # What lets a stack's matrices be gathered, tensor by tensor.
        stack = weight.shape[: self._STACK]
        ragged = [name for name in names if tensors[name].shape[: self._STACK] != stack]
        if ragged:
            raise ValueError(
                f"{type(self).__name__}: {', '.join(ragged)} does not open with "
                f"the weight's stack of {stack}"
            )
        self.mode, self.bits, self.group_size = mode, bits, group_size
        self.input_dims = input_dims
        for name in names:
            self[name] = tensors[name]
        if bias is not None:
            self["bias"] = bias
        self.freeze()

    @property
    def output_dims(self) -> int:
        return self["weight"].shape[-2]

    def _keep(self) -> tuple:
        """The layer's plan and bias, worked out and kept."""
        bias = self.get("bias")
        if "bias" in self and not isinstance(bias, mx.array):
            # MLX's quantized layers raise as they add it.
            raise TypeError(f"the bias is an array, not {type(bias).__name__}")
        kept = self.__dict__[_KEPT] = (_plan(self), bias, self["weight"])
        return kept

    def __setitem__(self, key: str, value: Any) -> None:
        self.__dict__.pop(_KEPT, None)
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        self.__dict__.pop(_KEPT, None)
        super().__delitem__(key)

    def pop(self, *key_and_default):
        self.__dict__.pop(_KEPT, None)
        return super().pop(*key_and_default)

    # A dict's own mutators, which do not go through the three above.
    def setdefault(self, key: str, default: Any = None) -> Any:
        self.__dict__.pop(_KEPT, None)
        return super().setdefault(key, default)

    def popitem(self) -> tuple:
        self.__dict__.pop(_KEPT, None)
        return super().popitem()

    def clear(self) -> None:
        self.__dict__.pop(_KEPT, None)
        super().clear()

    def __ior__(self, other):
        self.__dict__.pop(_KEPT, None)
        return super().__ior__(other)

    def __delattr__(self, name: str) -> None:
        self.__dict__.pop(_KEPT, None)
        super().__delattr__(name)

    def _decoded(self, indices: mx.array | None = None) -> mx.array:
        """The layer's weights as floats: all of them, or the matrices at
        ``indices`` of the stack."""
        fmt = formats.get(self.mode)
        tensors = {name: self[name] for name in ("weight", *fmt.tensors)}
        if indices is not None:
            tensors = {name: tensor[indices] for name, tensor in tensors.items()}
        return fmt.decode(tensors, self.bits, self.group_size)

    def _extra_repr(self) -> str:
        return (
            f"input_dims={self.input_dims}, output_dims={self.output_dims}, "
            f"bias={'bias' in self}, group_size={self.group_size}, "
            f"bits={self.bits}, mode={self.mode}"
        )


class FormatLinear(_FormatLayer):
    """``x @ W.T + bias`` for a matrix ``W`` kept in format ``mode``:
    ``tensors`` are its ``weight`` and the format's tensors, by name."""

    def __call__(self, x: mx.array) -> mx.array:
        shape, dtype = x.shape, x.dtype
        kept = self.__dict__.get(_KEPT)
        if kept is None or kept[2] is not self["weight"]:
            kept = self._keep()
        plan, bias, _ = kept
        # A decode step is this, its kept graph's replay and the bias.
        graph = plan.proof.graphs.get((shape, dtype)) if plan is not None else None
        if graph is not None:
            y = graph(x, *plan.operands)
        else:
            k = self.input_dims
            if not shape or shape[-1] != k:
                raise ValueError(f"x is (..., {k}), not of shape {shape}")
            y = None
            if plan is not None and 0 < x.size // k <= MAX_ROWS:
                y = _launch(plan, (shape, dtype), _linear, _linear_stand_in, x)
            if y is None:
                y = x @ self._decoded().astype(dtype).T
        return y if bias is None else y + bias


class FormatSwitchLinear(_FormatLayer):
    """A stack of matrices kept in format ``mode`` (a switch layer's
    experts), called as mlx-vlm's ``QuantizedSwitchLinear``: ``x`` is
    ``(..., M, K)``, ``indices`` name a matrix for each of ``x``'s leading
    axes, broadcast together, and the result is ``(..., M, N)``."""

    _STACK = 1

    @property
    def num_experts(self) -> int:
        return self["weight"].shape[0]

    def __call__(
        self, x: mx.array, indices: mx.array, sorted_indices: bool = False
    ) -> mx.array:
        # sorted_indices is mlx-vlm's hint to MLX's gathers. The rows kernel
        # runs every pair as one mat-vec, in any order; the decode passes the
        # hint on to MLX's gather_mm.
        shape, dtype = x.shape, x.dtype
        signature = (shape, dtype, indices.shape, indices.dtype)
        kept = self.__dict__.get(_KEPT)
        if kept is None or kept[2] is not self["weight"]:
            kept = self._keep()
        plan, bias, _ = kept
        graph = plan.proof.graphs.get(signature) if plan is not None else None
        if graph is not None:
            y = graph(x, indices, *plan.operands)
        else:
            k = self.input_dims
            if len(shape) < 2 or shape[-1] != k:
                raise ValueError(f"x is (..., M, {k}), not of shape {shape}")
            if not mx.issubdtype(indices.dtype, mx.integer):
                raise ValueError(f"indices are integers, not {indices.dtype}")
            y = _launch(plan, signature, _switch, _switch_stand_in, x, indices)
            if y is None and indices.size < self.num_experts:
                # A decode step's indices: their matrices alone are decoded.
                weights = self._decoded(indices).astype(dtype)
                y = x @ weights.swapaxes(-1, -2)
            elif y is None:
                # A prefill's: the stack once, not a matrix for each index.
                y = mx.gather_mm(
                    x,
                    self._decoded().astype(dtype).swapaxes(-1, -2),
                    rhs_indices=_wrapped(indices, self.num_experts),
                    sorted_indices=sorted_indices,
                )
        if bias is not None:
            y = y + mx.expand_dims(bias[indices], -2)
        return y
