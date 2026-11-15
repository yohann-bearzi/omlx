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
    return _Plan(face, _kernel(face, stacked), operands, proof, k, n)


def _runs(plan: _Plan | None, dtype) -> bool:
    return plan is not None and not plan.proof.failed and plan.face.exact_for(dtype)


def _rows(plan: _Plan, x, dtype, pairs: int, shape: tuple, rhs=None, per_row=1):
    """One mat-vec per pair, ``pairs`` of them, as ``shape`` (which ends
    with N). Pair ``p`` is row ``p // per_row`` of ``x``, rows of K
    ``dtype`` in any shape, with the matrix, or with matrix ``rhs[p]`` of
    the stack. No reshape is put in the graph for the kernel's sake."""
    n = plan.n
    inputs = [x, *plan.operands]
    template = [("T", dtype), ("K_SIZE", plan.k), ("N_SIZE", n)]
    if rhs is not None:
        inputs.append(rhs)
        template.append(("PAIRS_PER_ROW", per_row))
    return plan.kernel(
        inputs=inputs,
        template=template,
        grid=(32, 2 * pairs, n // 8),
        threadgroup=(32, 2, 1),
        output_shapes=[shape],
        output_dtypes=[dtype],
    )[0]


def _proven(plan: _Plan, launch, inputs: tuple, *rows):
    """``_rows(plan, *rows)``, or None when its launch does not compile.

    The launch is evaluated here the first time it is built, once ``inputs``
    and the layer's tensors are: what fails ahead of it is raised as it is."""
    proof = plan.proof
    if launch in proof.ran:
        return _rows(plan, *rows)
    mx.eval(*inputs, *plan.operands)
    y = _rows(plan, *rows)
    try:
        # Surface a kernel build failure while the call can still fall back.
        mx.eval(y)
    except Exception:
        proof.failed = True
        logger.warning(
            "quant format face %s: rows kernel failed; decode fallback",
            plan.face.key,
            exc_info=True,
        )
        return None
    proof.ran.add(launch)
    logger.info("quant format face %s: rows kernel engaged", plan.face.key)
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
        k, shape, dtype = self.input_dims, x.shape, x.dtype
        if shape[-1] != k:
            raise ValueError(f"x has {shape[-1]} features, the layer takes {k}")
        rows = x.size // k
        y = None
        if 0 < rows <= MAX_ROWS:
            plan = cached_per_module(self, "_omlx_format_plan", _plan, depth=0)
            if _runs(plan, dtype):
                y = _proven(plan, dtype, (x,), x, dtype, rows, (*shape[:-1], plan.n))
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
        k, shape, dtype = self.input_dims, x.shape, x.dtype
        if len(shape) < 2 or shape[-1] != k:
            raise ValueError(f"x is (..., M, {k}), not of shape {shape}")
        y = None
        plan = cached_per_module(self, "_omlx_format_plan", _plan, depth=0)
        if _runs(plan, dtype) and indices.size * x.size:
            lead, m, batch = shape[:-2], shape[-2], indices.shape
            own = len(lead)  # x's leading axes ahead of its trailing ones of 1
            while own and lead[own - 1] == 1:
                own -= 1
            if (
                m == 1
                and batch  # MLX binds indices of no axis as a scalar
                and len(batch) == len(lead)
                and batch[:own] == lead[:own]
            ):
                # Each row of x meets its own run of experts: nothing is
                # broadcast, in memory or in shape.
                rows, rhs, pairs = x, indices, indices.size
                per_row = pairs // (x.size // k)
            else:
                batch = mx.broadcast_shapes(lead, batch)
                rows = mx.broadcast_to(x, (*batch, m, k))
                rhs = mx.repeat(mx.broadcast_to(indices, batch), m)
                pairs, per_row = rhs.size, 1
            y = _proven(
                plan,
                # MLX binds fewer than 8 indices as constants: other Metal.
                (dtype, per_row, pairs < 8),
                (x, indices),
                rows,
                dtype,
                pairs,
                (*batch, m, plan.n),
                rhs.astype(mx.uint32),
                per_row,
            )
        if y is None:
            if indices.size < self.num_experts:
                # A decode step's indices: their matrices alone are decoded.
                weights = self._decoded(indices).astype(dtype)
                y = x @ weights.swapaxes(-1, -2)
            else:
                # A prefill's: the stack once, not a matrix for each index.
                y = mx.gather_mm(
                    x,
                    self._decoded().astype(dtype).swapaxes(-1, -2),
                    rhs_indices=indices,
                    sorted_indices=sorted_indices,
                )
        if "bias" in self:
            y = y + mx.expand_dims(self["bias"][indices], -2)
        return y
