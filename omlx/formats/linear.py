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
from ..patches.module_cache import cached_per_module
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
    operands: tuple  # the weight, then the face's tensors
    proof: kernels.Proof
    k: int
    n: int
    said: tuple[str, str]  # what a proof logs: the launches engaged, or failed


def _plan(layer) -> _Plan | None:
    """How ``layer``'s rows run through its format's dot face, or None: the
    format has no face for the matrix, or a launch could not take it."""
    k, n = layer.input_dims, layer.output_dims
    # MLX binds x as a buffer from 8 elements; a launch is whole tiles.
    if k < 8 or n % 8:
        return None
    face = formats.dot(layer, k, n)
    if face is None:
        return None
    operands = tuple(layer[name] for name in ("weight", *face.tensors))
    stacked = layer["weight"].ndim == 3
    # What a launch is compiled from, beside what the layer keeps in ran.
    proof = kernels.proof(
        "format_switch_linear" if stacked else "format_linear",
        face,
        tuple(a.dtype for a in operands),
        k,
        n,
    )
    said = (
        f"quant format face {face.key}: rows kernel engaged",
        f"quant format face {face.key}: rows kernel failed; decode fallback",
    )
    return _Plan(face, _kernel(face, stacked), operands, proof, k, n, said)


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


def _linear(kernel, k: int, n: int, x):
    """A matrix's launch for ``x``: what its Metal is instantiated with
    beside the proof's key, and its graph, from ``x`` and the operands.
    The graph holds no array: it outlives the layers it is built for."""
    pairs, shape = x.size // k, (*x.shape[:-1], n)

    def graph(x, *operands):
        return _rows(kernel, k, n, x, operands, pairs, shape)

    return x.dtype, graph


# MLX's shapes are int32: a buffer of zeros as one axis of bytes is smaller.
_ONE_BUFFER = 2**31


def _zeros(*specs) -> list[mx.array]:
    """Zeros of each ``(shape, dtype)`` of ``specs``, a stand-in's inputs,
    evaluated here, ahead of the proof and outside its guard: what fails in
    making them (an allocation, an error that an earlier evaluation left
    pending on the GPU) is not the face's. They are made on the GPU's
    stream, the launch's, which is synchronized after their evaluation, so
    such an error is raised here. They are views of one buffer of as many
    bytes as the largest, from its first byte: one fill and, where MLX's
    buffer cache is empty (oMLX clears it after each prefill chunk), one new
    buffer. Where the largest has 2 GiB or more, each is zeros of its own."""
    sizes = [math.prod(shape) * dtype.size for shape, dtype in specs]
    with mx.stream(mx.gpu):
        if max(sizes) < _ONE_BUFFER:
            buffer = mx.zeros((max(sizes),), mx.uint8)
            zeros = [
                mx.view(buffer[:size], dtype).reshape(shape)
                for (shape, dtype), size in zip(specs, sizes, strict=True)
            ]
        else:
            zeros = [mx.zeros(shape, dtype) for shape, dtype in specs]
        mx.eval(*zeros)
        mx.synchronize(mx.gpu)
    return zeros


def _linear_stand_in(plan: _Plan, metal) -> mx.array:
    """The smallest launch of a matrix that MLX generates the Metal of
    ``metal`` for (x's type), over zeros: one vector, and the whole of each
    tensor. What follows the rows in a matrix's tensors is the format's
    own, so no part of a tensor is known to be all that a row reads."""
    operands = ((a.shape, a.dtype) for a in plan.operands)
    x, *tensors = _zeros(((plan.k,), metal), *operands)
    # Its first tile of outputs.
    return _rows(plan.kernel, plan.k, plan.n, x, tensors, 1, (plan.n,), None, 1, 1)


def _switch(kernel, k: int, n: int, x, indices):
    """A stack's launch for ``x`` and ``indices``, as ``_linear`` gives a
    matrix's, with the indices ahead of the operands; None when there is no
    pair to launch, or more than a grid takes. A negative index counts from
    the stack's end, as MLX's indexing does."""
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
        per_row = pairs // (x.size // k)
    else:
        batch = mx.broadcast_shapes(lead, batch)
        pairs, per_row = math.prod(batch) * m, 1
    # MLX's grid is int32, of 2 rows of threadgroups a pair.
    if pairs >= 2**30:
        return None
    shape = (*batch, m, n)

    def graph(x, indices, *operands):
        if not as_they_are:
            x = mx.broadcast_to(x, (*batch, m, k))
            indices = mx.repeat(mx.broadcast_to(indices, batch), m)
        rhs = _wrapped(indices, operands[0].shape[0]).astype(mx.uint32)
        return _rows(kernel, k, n, x, operands, pairs, shape, rhs, per_row)

    # MLX binds fewer than 8 indices as constants: other Metal.
    return (x.dtype, per_row, pairs < 8), graph


def _wrapped(indices, matrices: int):
    """``indices`` with each negative one counted from the end of a stack of
    ``matrices``."""
    if not mx.issubdtype(indices.dtype, mx.signedinteger):
        return indices
    return mx.where(indices < 0, indices + matrices, indices)


def _switch_stand_in(plan: _Plan, metal) -> mx.array:
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
    x, *tensors, rhs = _zeros(((rows, plan.k), dtype), *operands, ((pairs,), mx.uint32))
    shape = (pairs, plan.n)
    # Each pair's first tile of outputs.
    return _rows(plan.kernel, plan.k, plan.n, x, tensors, pairs, shape, rhs, per_row, 1)


def _launch(plan: _Plan | None, signature: tuple, describe, stand_in, *arrays):
    """``arrays`` (x, then a stack's indices) through the rows kernel, the
    first time for ``signature``, or None: the layer has no plan, its proof
    has failed, the face is not exact for x's type, there is nothing to
    launch, or the launch does not compile.

    The launch's graph (``describe``) is compiled by MLX and kept under
    ``signature``, which is what it depends on (the arrays' shapes and x's
    type), by the thread that builds it: a layer's call asks for the graph
    its thread kept first, and replays it.

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
    launch = describe(plan.kernel, plan.k, plan.n, *arrays)
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


class _FormatLayer(nn.Module):
    """What the two layers share: a format's tensors as frozen parameters,
    and the attributes of MLX's quantized layers."""

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
        plan = cached_per_module(self, "_omlx_format_plan", _plan, depth=0)
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
        if "bias" in self:
            y = y + self["bias"]
        return y


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
        plan = cached_per_module(self, "_omlx_format_plan", _plan, depth=0)
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
        if "bias" in self:
            y = y + mx.expand_dims(self["bias"][indices], -2)
        return y
