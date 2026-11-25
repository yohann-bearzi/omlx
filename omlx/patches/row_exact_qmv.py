# SPDX-License-Identifier: Apache-2.0
"""Multi-row quantized projection with one-row decode arithmetic per row.

Stock ``quantized_matmul`` switches kernels with the row count (``qmv_fast``
or ``qmv`` at one row, ``qmv_wide`` and then tiled qmm for more), so a
speculative verify row and the serial decode step for the same token get
different bits from the same projection. Here every row runs MLX 0.32.4's
one-row ``qmv_fast`` (K of the kernel block: 512 for
4/5-bit, 256 for 6/8-bit weights) or ``qmv`` traversal, transcribed in
``omlx.formats.affine``, so row ``r`` of the output equals
``quantized_matmul(x[r:r+1])`` bit for bit.

A threadgroup computes one ``2 * RPS``-column output tile for ``ROWS`` rows:
each weight block is decoded once and applied to all of them, and every
(row, column) accumulator still sums its K blocks in qmv order and finishes
with the same ``simd_sum`` (qmv's lane layout per column does not depend on
how many columns a simdgroup owns or how many rows share the threadgroup).

Both stock MLX affine layers and ``FormatLinear`` layers route through the
unified ``omlx.formats.row_exact_skeleton`` (``RowExactPlan`` built once at
construction / first signature); one launch may run several projections of
one input.
"""

from __future__ import annotations

import logging
import threading
import weakref
from dataclasses import replace
from functools import cache

import mlx.core as mx
import mlx.nn as nn

from .. import formats
from ..formats import Dot, affine, kernels, row_exact_skeleton, tuning
from ..formats.affine import _GROUP_SIZES, qmv_fast_layout, values_per_thread
from ..formats.linear import FormatLinear
from ..formats.row_exact_skeleton import RowExactPlan
from . import module_cache
from .module_cache import cached_per_module

logger = logging.getLogger(__name__)

MAX_ROWS = 16
# The affine widths the row-exact verify tile decodes. 2- and 3-bit stock
# layers keep MLX's path here, as they did in upstream's verify gather
# (``_BITS`` 4/5/6/8 too): they are out of the PoC's served widths, not
# refused by ``affine.decoded`` (which takes them).
_BITS = (4, 5, 6, 8)
# Large 6- and 8-bit heads can amortize padding over a shared weight pass.
_PADDED_ROWS_MIN_WEIGHT_BYTES = 128 * 1024 * 1024

# The envelope of kernels.unrolled_tile_ok was measured on one GPU
# generation; another compiler or GPU can still get a tile inside it wrong (an
# M3 Ultra did at 4-bit gs64: whole rows off, not rounding). So the first
# launch of each unrolled shape is checked on the running GPU against stock
# one-row ``quantized_matmul`` of the same weights on a fixed input; a shape
# with any differing bit runs the rolled tile (same arithmetic, same
# geometry) from then on. Verdicts are per compiled shape: the bug is in code
# generation, not in the weight values.
_UNROLLED_VERDICTS: dict = {}
_UNROLLED_LOCKS_GUARD = threading.Lock()
_UNROLLED_LOCKS: dict = {}


def unrolled_tile_verified(
    shape: tuple, run, reference, rows: int, k: int, dtype
) -> bool:
    """Whether the unrolled launch ``shape`` (a hashable key of every template
    parameter) gives the one-row calls' bits. ``run(x)`` launches it on ``x``
    [rows, k] and ``reference(x)`` makes each of its outputs one row at a
    time; checked once per shape."""
    verdict = _UNROLLED_VERDICTS.get(shape)
    if verdict is None:
        with _UNROLLED_LOCKS_GUARD:
            lock = _UNROLLED_LOCKS.setdefault(shape, threading.Lock())
        with lock:
            verdict = _UNROLLED_VERDICTS.get(shape)
            if verdict is None:
                x = (mx.random.normal((rows, k), key=mx.random.key(rows)) * 0.5).astype(
                    dtype
                )
                got, want = run(x), reference(x)
                mx.eval(got, want)
                verdict = all(
                    mx.array_equal(a.reshape(b.shape), b).item()
                    for a, b in zip(got, want)
                )
                verdict = _UNROLLED_VERDICTS.setdefault(shape, verdict)
                if not verdict:
                    logger.warning(
                        "row-exact unrolled tile %s differs from the one-row call on "
                        "this GPU; it runs the rolled tile instead",
                        shape,
                    )
    return verdict


def _stock_rows(weights: list, bits: int, group_size: int):
    """The reference of a launch over affine tensors: stock one-row
    ``quantized_matmul`` of each ``(weight, scales, biases)``, row by row."""

    def reference(x: mx.array) -> list:
        return [
            mx.concatenate(
                [
                    mx.quantized_matmul(
                        x[r : r + 1],
                        w,
                        scales=s,
                        biases=b,
                        transpose=True,
                        group_size=group_size,
                        bits=bits,
                    )
                    for r in range(x.shape[0])
                ]
            )
            for w, s, b in weights
        ]

    return reference


def _face(linear, x: mx.array) -> Dot | None:
    """The dot face (``formats.dot``) ``linear``'s rows run over, or None.
    ``linear`` is an MLX affine layer of this module's widths (a subclass
    too: the prefill reclass makes these projections one) or a format layer,
    and its face is exact for ``x``'s dtype and reads K in whole lanes."""
    if isinstance(linear, nn.QuantizedLinear):
        if getattr(linear, "mode", "affine") != "affine" or linear.bits not in _BITS:
            return None
    elif not isinstance(linear, FormatLinear) or x.shape[-1] != linear.input_dims:
        return None
    k = int(x.shape[-1])
    # MLX binds x as a constant below 8 elements; the tile reads it from
    # device memory (linear.py's _plan refuses the same).
    if k < 8:
        return None
    if formats.input_transform(linear) is not None:
        # The row-exact kernels read the activation as it arrives: a format
        # that rotates it must reach them through its own prepared rows.
        return None
    face = formats.dot(linear, k, int(linear[formats.tensor_names(linear)[0]].shape[0]))
    if face is None or not face.exact_for(x.dtype) or face.vpt < 1 or k % face.vpt:
        return None
    return face


def _operands(linear, face: Dot) -> tuple:
    return (
        linear[formats.tensor_names(linear)[0]],
        *(linear[name] for name in face.tensors),
    )


def _padded(bits: int, fast: bool, n: int, k: int) -> bool:
    """Whether to pad a partial tile on an affine head's fast traversal."""
    return (
        bits in (6, 8) and fast and n * k * bits // 8 >= _PADDED_ROWS_MIN_WEIGHT_BYTES
    )


def _unrolled_ok(bits: int, rps: int, rows_per_group: int) -> bool:
    """kernels.unrolled_tile_ok for the qmv_fast tile of ``bits``-bit weights."""
    return kernels.unrolled_tile_ok(values_per_thread(bits, True), rps, rows_per_group)


def _resolve_geometry(
    linears,
    faces,
    sizes: tuple[int, ...],
    k: int,
    rows: int,
    geometry,
    *,
    grouped: bool,
) -> tuple[int, int, bool] | None:
    """Resolve ``(rows_per_group, rps, unrolled)`` for ``linears`` at ``rows``."""
    vpt = faces[0].vpt
    if geometry is None:
        first = linears[0]
        bits = int(first.bits)
        policy = tuning.row_exact_policy(
            "group" if grouped else "linear",
            rows,
            bits,
            k,
            max(sizes),
            kernels.gpu_generation(),
        )
        if policy is not None:
            per_group, rps, unrolled = (
                policy.rows_per_group,
                policy.rps,
                policy.unrolled,
            )
        else:
            # The padding rule is affine's: a format layer of the built-in
            # mode keeps it.
            pad = getattr(first, "mode", "affine") == "affine" and _padded(
                bits, qmv_fast_layout(k, max(sizes), bits), max(sizes), k
            )
            per_group, rps = kernels.shared_geometry(vpt, max(sizes), rows, pad)
            if bits == 6 and rows % per_group:
                # Three inputs share a 6-bit decode. Four pads a five-row
                # window to eight and regresses the 27B head on the M5 Max.
                per_group = 3
            unrolled = False
    else:
        per_group, rps, unrolled = geometry
        if per_group < 1 or rps < 1 or rows % per_group:
            return None
    unrolled = unrolled and kernels.unrolled_tile_ok(vpt, rps, per_group)
    return per_group, rps, unrolled


def _module_launch(
    linears, faces, x: mx.array, rows: int, geometry=None, *, grouped: bool = True
) -> RowExactPlan | None:
    """Build a ``RowExactPlan`` for ``linears`` (layers over ``faces``) at
    ``rows`` rows of ``x``, on ``geometry`` ``(rows_per_group, rps, unrolled)``
    or the default policy; None where ``geometry`` does not fit or a plugin
    face fails cold proof."""
    k = int(x.shape[-1])
    names = tuple(formats.tensor_names(linear)[0] for linear in linears)
    operand_keys = tuple((name, *face.tensors) for name, face in zip(names, faces))
    operands = tuple(
        linear[key] for linear, keys in zip(linears, operand_keys) for key in keys
    )
    sizes = tuple(int(linear[name].shape[0]) for linear, name in zip(linears, names))
    resolved = _resolve_geometry(
        linears, faces, sizes, k, rows, geometry, grouped=grouped
    )
    if resolved is None:
        return None
    per_group, rps, unrolled = resolved
    plan = row_exact_skeleton.build_plan(
        faces,
        sizes,
        k,
        rows,
        x.dtype,
        (per_group, rps, unrolled),
        operands=operands,
        names=names,
        operand_keys=operand_keys,
        grouped=grouped,
    )
    if plan is None:
        return None
    if plan.shape is not None:

        def run(probe):
            outputs = plan(probe)
            return [_biased(m, y) for m, y in zip(linears, outputs)]

        def reference(probe):
            return [_one_row_calls(linear, probe) for linear in linears]

        if not unrolled_tile_verified(plan.shape, run, reference, rows, k, x.dtype):
            plan = row_exact_skeleton.build_plan(
                faces,
                sizes,
                k,
                rows,
                x.dtype,
                (per_group, rps, False),
                operands=operands,
                names=names,
                operand_keys=operand_keys,
                grouped=grouped,
            )
    return plan


def _format_plans(linear):
    # A checkpoint reader chooses its face from the layer's actual parameters.
    # Other projections in a group keep the token, without retaining the arrays.
    return cached_per_module(
        linear,
        "_omlx_format_row_exact_plans",
        lambda _: ({}, object()),
        depth=0,
        flags=(
            linear.fmt,
            linear.input_dims,
            type(linear.bits),
            linear.bits,
            type(linear.group_size),
            linear.group_size,
            tuple(dict.keys(linear)),
        ),
    )


_GROUP_PLANS_LOCK = threading.Lock()
_MISS = object()
_DICT_GET = dict.get
_PLANS_SLOT = "_omlx_row_exact_plans"
_DECLINE_KEYS = ("weight", "scales", "biases", "bias")


class _Declined:
    """A cached refusal of one or more projections, with the tensors and bias
    keys it was made under: the cache answers None only while they are still
    the module's, and a build follows any of them changing."""

    __slots__ = ("checks",)

    def __init__(self, checks: tuple) -> None:
        self.checks = checks

    def matches(self, linears) -> bool:
        dget = _DICT_GET
        for idx, key, arr, present in self.checks:
            if (key in linears[idx]) != present or dget(linears[idx], key) is not arr:
                return False
        return True


def _declined(linears) -> _Declined:
    """The refusal cache of ``linears``' current tensors and bias keys."""
    dget = _DICT_GET
    return _Declined(
        tuple(
            (idx, key, dget(linear, key), key in linear)
            for idx, linear in enumerate(linears)
            for key in _DECLINE_KEYS
        )
    )


def _plans(linear) -> dict:
    plans = linear.__dict__.get(_PLANS_SLOT)
    if plans is not None:
        return plans
    if isinstance(linear, FormatLinear):
        return _format_plans(linear)[0]
    # Plans live on the (first) module, so they die with it; operand identities
    # are checked lock-free on warm hits so weight writes refresh the plan.
    # Registered, so ``clear_cached`` drops them with the tensors they hold.
    module_cache.register_slot(_PLANS_SLOT)
    return linear.__dict__.setdefault(_PLANS_SLOT, {})


def _plan(linear, x: mx.array, rows: int) -> RowExactPlan | None:
    if not (1 <= rows <= MAX_ROWS):
        return None
    plans = _plans(linear)
    key = (rows, x.dtype, x.shape[-1])
    cached = plans.get(key, _MISS)
    if cached is not _MISS:
        if isinstance(cached, _Declined):
            if cached.matches((linear,)):
                return None
        else:
            dget = _DICT_GET
            for name, arr in cached.single_checks:
                if dget(linear, name) is not arr:
                    break
            else:
                return cached
    declined = _declined((linear,))
    face = _face(linear, x)
    built = (
        None
        if face is None
        else _module_launch((linear,), (face,), x, rows, grouped=False)
    )
    existing = plans.get(key, _MISS)
    if isinstance(existing, RowExactPlan):
        dget = _DICT_GET
        for name, arr in existing.single_checks:
            if dget(linear, name) is not arr:
                break
        else:
            return existing
    plans[key] = built if built is not None else declined
    return built


def _one_row_calls(linear, x: mx.array) -> mx.array:
    """``linear`` on each row of ``x`` ([rows, K]) alone: the decode call."""
    return mx.concatenate([linear(x[r : r + 1][None])[0] for r in range(x.shape[0])])


def _biased(linear, y: mx.array) -> mx.array:
    bias = _DICT_GET(linear, "bias")
    return y + bias if bias is not None else y


def quantized_linear(linear, x: mx.array) -> mx.array:
    """``linear(x)`` for ``x`` of shape [..., K], each row with one-row arithmetic.

    ``linear`` is an MLX quantized layer or a format layer. Layers the kernel
    does not take (no dot face for ``x``, other modes or bit widths, more
    than MAX_ROWS rows) run one one-row call per row, which is the serial
    decode call itself.
    """
    shape = x.shape
    k = shape[-1]
    rows = x.size // k if k else 0
    if rows <= 1:
        return linear(x)
    dget = _DICT_GET
    plans = linear.__dict__.get(_PLANS_SLOT)
    if plans is not None:
        plan = plans.get((rows, x.dtype, k), _MISS)
        if plan is _MISS or isinstance(plan, _Declined):
            plan = _plan(linear, x, rows)
        else:
            for name, arr in plan.single_checks:
                if dget(linear, name) is not arr:
                    plan = _plan(linear, x, rows)
                    break
    else:
        plan = _plan(linear, x, rows)
    flat = x if len(shape) == 2 else x.reshape(rows, k)
    if plan is None:
        y = _one_row_calls(linear, flat)
        return y if len(shape) == 2 else y.reshape(*shape[:-1], -1)
    y = plan(flat)[0]
    bias = dget(linear, "bias")
    if bias is not None:
        y = y + bias
    return y if len(shape) == 2 else y.reshape(*shape[:-1], -1)


def _group_plan(linears, x: mx.array, rows: int, geometry=None) -> RowExactPlan | None:
    count = len(linears)
    if not count or not (1 <= rows <= MAX_ROWS):
        return None
    first = linears[0]
    plans = _plans(first)
    dget = _DICT_GET
    key = (count, rows, x.dtype, x.shape[-1], geometry)
    if count == 1:
        cached = plans.get(key, _MISS)
        if cached is not _MISS:
            if isinstance(cached, _Declined):
                if cached.matches(linears):
                    return None
            elif "bias" not in first:
                for name, arr in cached.single_checks:
                    if dget(first, name) is not arr:
                        break
                else:
                    return cached
        declined = _declined(linears)
        face = _face(first, x)
        launch = (
            _module_launch(linears, (face,), x, rows, geometry, grouped=True)
            if (
                face is not None
                and "bias" not in first
                and kernels.table_bytes(face) <= formats.TABLE_BYTES
            )
            else None
        )
        with _GROUP_PLANS_LOCK:
            existing = plans.get(key, _MISS)
            if isinstance(existing, RowExactPlan) and launch is not None:
                if "bias" not in first:
                    for name, arr in existing.single_checks:
                        if dget(first, name) is not arr:
                            break
                    else:
                        return existing
            elif (
                isinstance(existing, _Declined)
                and launch is None
                and existing.matches(linears)
            ):
                return None
            plans[key] = launch if launch is not None else declined
        return launch
    # The members after the first are keyed by their identity, not their
    # address: a weak reference to each, checked on every hit, so a freed
    # member's address reused by another layer cannot return its plan.
    # Entries are stored as immutable tuples so warm hits read lock-free
    # without allocating weakrefs, lists or generators; updates serialize
    # under _GROUP_PLANS_LOCK outside _module_launch. Format siblings are
    # keyed by the tokens they had when the entry was built, so a warm hit
    # compares stored tokens instead of re-slicing the members.
    entries = plans.get(key)
    if entries is not None:
        for entry_refs, entry_fmt_checks, entry_launch in entries:
            matched = True
            for i, ref in enumerate(entry_refs, 1):
                if ref() is not linears[i]:
                    matched = False
                    break
            if not matched:
                continue
            for i, entry_token in entry_fmt_checks:
                if _format_plans(linears[i])[1] is not entry_token:
                    matched = False
                    break
            if not matched:
                break
            if isinstance(entry_launch, _Declined):
                if entry_launch.matches(linears):
                    return None
                break
            launch = entry_launch
            ok = True
            for m in linears:
                if "bias" in m:
                    ok = False
                    break
            if ok:
                for idx, name, arr in launch.group_checks:
                    if dget(linears[idx], name) is not arr:
                        ok = False
                        break
            if ok:
                return launch
            break
    members = linears[1:]
    fmt_checks = tuple(
        (i, _format_plans(linear)[1])
        for i, linear in enumerate(members, 1)
        if isinstance(linear, FormatLinear)
    )
    declined = _declined(linears)
    faces = [_face(linear, x) for linear in linears]
    grouped = (
        faces
        and all(face is not None for face in faces)
        and len({face.vpt for face in faces}) == 1
        and not any("bias" in linear for linear in linears)
        and kernels.table_bytes(*faces) <= formats.TABLE_BYTES
    )
    launch = (
        _module_launch(linears, faces, x, rows, geometry, grouped=True)
        if grouped
        else None
    )
    refs = tuple(weakref.ref(linear) for linear in members)
    stored = launch if launch is not None else declined
    with _GROUP_PLANS_LOCK:
        kept = []
        for entry in plans.get(key, ()):
            entry_refs, entry_fmt_checks, entry_launch = entry
            if not all(ref() is not None for ref in entry_refs):
                continue
            if all(ref() is linear for ref, linear in zip(entry_refs, members)):
                if entry_fmt_checks == fmt_checks:
                    if (
                        launch is None
                        and isinstance(entry_launch, _Declined)
                        and entry_launch.matches(linears)
                    ):
                        return None
                    elif (
                        launch is not None
                        and isinstance(entry_launch, RowExactPlan)
                        and entry_launch.operands_match(linears)
                        and not any("bias" in linear for linear in linears)
                    ):
                        return entry_launch
                continue
            kept.append(entry)
        kept.append((refs, fmt_checks, stored))
        plans[key] = tuple(kept)
    return launch


def _launch_group(plan: RowExactPlan, x: mx.array, rows: int, shape: tuple) -> tuple:
    flat = x if len(shape) == 2 else x.reshape(rows, shape[-1])
    outputs = plan(flat)
    if len(shape) == 2:
        return tuple(outputs)
    lead = shape[:-1]
    if len(outputs) == 1:
        return (outputs[0].reshape(*lead, -1),)
    return tuple([y.reshape(*lead, -1) for y in outputs])


def quantized_linears(linears, x: mx.array) -> tuple:
    """``tuple(linear(x) for linear in linears)`` with one-row arithmetic per
    row, in a single launch when the projections have faces of one vpt and
    no bias, and their distinct tables fit in one launch."""
    shape = x.shape
    k = shape[-1]
    rows = x.size // k if k else 0
    plan = _group_plan(linears, x, rows) if rows > 1 and 1 < len(linears) <= 4 else None
    if plan is None:
        return tuple(quantized_linear(linear, x) for linear in linears)
    return _launch_group(plan, x, rows, shape)


def quantized_linears_tiled(
    linears, x: mx.array, rows_per_group: int, rps: int, unrolled: bool = False
):
    """``quantized_linears`` in one launch for 1..MAX_ROWS rows on an explicit
    tile: each threadgroup applies ``2 * rps`` output columns to
    ``rows_per_group`` rows, on the unrolled tile if ``unrolled`` (inside
    ``kernels.unrolled_tile_ok``). Every tile gives each row its one-row bits
    (at one row, the layer's own call's); None where the launch does not
    take these projections."""
    shape = x.shape
    k = shape[-1]
    rows = x.size // k if k else 0
    geometry = (rows_per_group, rps, unrolled)
    if not isinstance(linears, tuple):
        linears = tuple(linears)
    if len(linears) == 1:
        first = linears[0]
        plans = first.__dict__.get(_PLANS_SLOT)
        if plans is not None:
            plan = plans.get((1, rows, x.dtype, k, geometry), _MISS)
            if plan is _MISS or isinstance(plan, _Declined):
                plan = _group_plan(linears, x, rows, geometry)
            elif plan is not None:
                dget = _DICT_GET
                if "bias" in first:
                    plan = _group_plan(linears, x, rows, geometry)
                else:
                    for name, arr in plan.single_checks:
                        if dget(first, name) is not arr:
                            plan = _group_plan(linears, x, rows, geometry)
                            break
        else:
            plan = _group_plan(linears, x, rows, geometry)
    else:
        plan = _group_plan(linears, x, rows, geometry)
    if plan is None:
        return None
    return _launch_group(plan, x, rows, shape)


def format_rows_qmv(linear, dtype, geometry):
    """Plan a format matrix on the GDN's (columns, group) tile for each width."""
    k = linear.input_dims
    probe = mx.zeros((1, k), dtype)
    face = _face(linear, probe)
    if face is None or kernels.table_bytes(face) > formats.TABLE_BYTES:
        return None
    operands = _operands(linear, face)
    operand_dtypes = tuple(a.dtype for a in operands)
    n = linear.output_dims
    if affine.row_ops_spec(face) is None:
        operand_specs = tuple((a.shape, a.dtype) for a in operands)
        if not row_exact_skeleton.prove_dot_face(face, k, n, dtype, operand_specs):
            return None

    @cache
    def tile(rows):
        rps, group = geometry(rows)
        plan = _module_launch(
            (linear,), (face,), probe, rows, (group, rps, rows > 1), grouped=False
        )
        if plan is None:
            raise ValueError(f"no {rps}x{group} tile for {rows} rows of {n}")
        # The plan binds the face, geometry and verified unroll choice; the
        # proof key carries them too, so graphs of one width or tile are never
        # replayed for another's launch.
        proof = kernels.proof(
            "format_gdn_rows",
            plan.kernel,
            tuple(plan.template),
            plan.grid,
            plan.padding,
            dtype,
            operand_dtypes,
        )
        return plan, proof

    def project(x):
        shape = x.shape
        if not shape or shape[-1] != k or x.dtype != dtype:
            raise ValueError(
                f"row-exact tile: expected [..., {k}] {dtype}, got {shape} {x.dtype}"
            )
        rows = x.size // k
        plan, proof = tile(rows)
        signature = (shape, dtype)
        graph = proof.graphs.get(signature)
        if graph is not None:
            return graph(x, *operands)

        # Equal tiles share the graph on this thread, each with its own weights;
        # the kept graph closes over an operand-free plan, so a proof outliving
        # the project holds none of the layer's arrays.
        graph_plan = replace(plan, operands=(), single_checks=(), group_checks=())

        @mx.compile
        def graph(value, *tensors):
            (y,) = graph_plan(value.reshape(rows, k), tensors)
            return y.reshape(*value.shape[:-1], n)

        y = graph(x, *operands)
        proof.keep(signature, graph)
        return y

    return project


def _affine_face(bits: int, group_size: int, dtype) -> Dot:
    """The ``qmv_fast`` face of affine tensors taken by position, of scales
    and activations of ``dtype`` (float32 too: GLM's)."""
    if bits not in _BITS:
        raise ValueError(f"row-exact tile: no decode of {bits}-bit weights")
    return affine.face(bits, group_size, True, dtype)  # refuses a group size MLX lacks


def _qmv_fast_layout(
    weight, scales, biases, bits, group_size, mode, dtype, rps
) -> bool:
    """Stock one-row ``quantized_matmul`` runs ``qmv_fast`` on this layout
    (``qmv_fast_layout``) and ``2 * rps`` divides N."""
    if (
        mode != "affine"
        or bits not in _BITS
        or group_size not in _GROUP_SIZES
        or biases is None
        or dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or weight.dtype != mx.uint32
        or weight.ndim != 2
        or scales.dtype != dtype
        or biases.dtype != dtype
        or scales.shape != biases.shape
    ):
        return False
    n = int(weight.shape[0])
    k = int(scales.shape[-1]) * group_size
    return not (
        not qmv_fast_layout(k, n, bits)
        or n % (2 * rps)
        or scales.shape != (n, k // group_size)
        or weight.shape[1] * 32 != k * bits
        # MLX binds a tensor of fewer than 8 elements as a constant; the tile
        # reads scales and biases from device memory.
        or scales.size < 8
    )


class RowsQmv:
    """Row-exact affine projection for a block of rows: every row of ``x`` gets one-row
    ``qmv_fast`` bits. ``geometry(rows)`` gives the output columns per
    simdgroup and the rows each threadgroup applies its decoded weight tile
    to (a divisor of ``rows``); launch parameters are kept per row count.
    ``unrolled`` runs the fully unrolled tile (same bits) on tiles inside
    ``kernels.unrolled_tile_ok`` that this GPU computes bit-exactly (checked once per
    tile shape), else the rolled tile."""

    __slots__ = (
        "_face",
        "_weights",
        "_k",
        "_n",
        "_dtype",
        "_geometry",
        "_launch",
        "_bits",
        "_group_size",
        "_unrolled",
    )

    def __init__(
        self, weight, scales, biases, bits, group_size, dtype, geometry, unrolled=False
    ):
        self._face = _affine_face(bits, group_size, dtype)
        self._weights = (weight, scales, biases)
        self._k = int(scales.shape[-1]) * group_size
        self._n = int(weight.shape[0])
        self._dtype = dtype
        self._geometry = geometry
        self._launch = {}
        self._bits = bits
        self._group_size = group_size
        self._unrolled = unrolled

    def __call__(self, x: mx.array) -> mx.array:
        """``x`` is ``[..., K]`` of the planned dtype, row-contiguous."""
        shape = x.shape
        if not shape or shape[-1] != self._k or x.dtype != self._dtype:
            raise ValueError(
                f"row-exact tile: expected [..., {self._k}] {self._dtype}, "
                f"got {shape} {x.dtype}"
            )
        rows = x.size // self._k
        plan = self._launch.get(rows)
        if plan is None:
            rps, per_group = self._geometry(rows)
            if (
                rows < 1
                or per_group < 1
                or rps < 1
                or rows % per_group
                or self._n % (2 * rps)
                or (self._unrolled and not _unrolled_ok(self._bits, rps, per_group))
            ):
                raise ValueError(
                    f"no {rps}x{per_group} tile for {rows} rows of {self._n}"
                )
            plan = row_exact_skeleton.build_plan(
                (self._face,),
                (self._n,),
                self._k,
                rows,
                self._dtype,
                (per_group, rps, self._unrolled),
                operands=self._weights,
                names=("weight",),
                operand_keys=(("weight", "scales", "biases"),),
                grouped=False,
                direct_input=True,
            )
            if plan is None:
                raise ValueError(
                    f"no {rps}x{per_group} tile for {rows} rows of {self._n}"
                )
            if plan.shape is not None and not unrolled_tile_verified(
                plan.shape,
                lambda probe: plan(probe),
                _stock_rows([self._weights], self._bits, self._group_size),
                rows,
                self._k,
                self._dtype,
            ):
                plan = row_exact_skeleton.build_plan(
                    (self._face,),
                    (self._n,),
                    self._k,
                    rows,
                    self._dtype,
                    (per_group, rps, False),
                    operands=self._weights,
                    names=("weight",),
                    operand_keys=(("weight", "scales", "biases"),),
                    grouped=False,
                    direct_input=True,
                )
                if plan is None:
                    raise ValueError(
                        f"no {rps}x{per_group} tile for {rows} rows of {self._n}"
                    )
            plan = self._launch.setdefault(rows, plan)
        return plan(x)[0]


def rows_qmv(
    weight, scales, biases, bits, group_size, mode, dtype, geometry, unrolled=False
):
    """A ``RowsQmv`` on a ``_qmv_fast_layout`` (checked at one column per
    simdgroup; ``__call__`` checks the tile ``geometry`` picks), else None."""
    if not _qmv_fast_layout(weight, scales, biases, bits, group_size, mode, dtype, 1):
        return None
    return RowsQmv(weight, scales, biases, bits, group_size, dtype, geometry, unrolled)


_QUANTIZED_TYPES = (FormatLinear, nn.QuantizedLinear)
_VERIFY_WARM_ROWS = (2, 3, 4)


def _fused_attn_tile(rows: int) -> tuple[int, int, bool]:
    """Matches ``qwen4_exp/language.py:_fused_tile`` without importing it eagerly."""
    if rows == 1:
        return 1, 2, False
    return (2, 4, True) if rows % 2 == 0 else (1, 4, True)


def _linear_in_out(layer) -> tuple[int, int] | None:
    if isinstance(layer, FormatLinear):
        return layer.input_dims, layer.output_dims
    if isinstance(layer, nn.QuantizedLinear):
        w = layer.get("weight")
        if not isinstance(w, mx.array) or w.ndim != 2 or not layer.bits:
            return None
        return int(w.shape[1]) * 32 // int(layer.bits), int(w.shape[0])
    return None


def _linear_sig(layer) -> tuple | None:
    dims = _linear_in_out(layer)
    if dims is None:
        return None
    bits = layer.bits if type(layer.bits) is int else None
    group_size = layer.group_size if type(layer.group_size) is int else None
    return (
        type(layer).__name__,
        getattr(layer, "mode", "affine"),
        bits,
        group_size,
        dims[0],
        dims[1],
        "bias" in layer,
    )


def _plan_stand_in(
    plan: RowExactPlan, k: int, dtype: mx.Dtype
) -> tuple[tuple[tuple[tuple[int, ...], mx.Dtype], ...], kernels.StandIn]:
    """Minimal safe stand-in specs and StandIn for one tile of ``plan``.

    Launches a single threadgroup ``(tile=0, row0=0)`` reading rows ``0..ROWS-1``
    of ``x`` and columns ``0..min(N_i, 2 * RPS)-1`` of projection 0's tensors
    while preserving MLX's ``size < 8`` / ``0-d`` constant-buffer binding.
    Holds zero ``mx.array`` references.
    """
    tpl = dict(plan.template)
    per_group = int(tpl["ROWS"])
    rps = int(tpl["RPS"])
    op_specs: list[tuple[tuple[int, ...], mx.Dtype]] = [((per_group, k), dtype)]
    pos = 0
    for out_shape, keys in zip(plan.output_shapes, plan.operand_keys):
        n_j = int(out_shape[-1])
        for _ in keys:
            arr = plan.operands[pos]
            pos += 1
            if arr.ndim >= 1 and arr.shape[0] == n_j:
                if arr.size >= 8:
                    min_rows = max(2 * rps, -(-8 * n_j // arr.size))
                    stand_n = min(n_j, min_rows)
                    shape = (stand_n, *arr.shape[1:])
                else:
                    shape = tuple(arr.shape)
            else:
                shape = tuple(arr.shape)
            op_specs.append((shape, arr.dtype))
    in_specs = tuple(op_specs)
    in_shapes = tuple(s for s, _ in in_specs)
    out_shapes = tuple((per_group, int(s[-1])) for s in plan.output_shapes)
    stand_in = kernels.StandIn(
        input_shapes=in_shapes,
        output_shapes=out_shapes,
        grid=(32, 2, 1),
        threadgroup=(32, 2, 1),
    )
    return in_specs, stand_in


def _register_rolled_instance(
    plan: RowExactPlan, k: int, dtype: mx.Dtype, instances: dict
) -> None:
    in_specs, stand_in = _plan_stand_in(plan, k, dtype)
    inst = kernels.planned(
        plan.kernel,
        in_specs,
        plan.output_dtypes,
        template=plan.template,
    )
    instances.setdefault(inst, lambda si=stand_in: si)


def _plan_warm_target(
    linears: tuple,
    k: int,
    rows: int,
    dtype: mx.Dtype,
    geometry,
    *,
    grouped: bool,
    instances: dict,
    unrolled_tasks: dict,
    ref_proof_tasks: dict,
) -> None:
    """Register load-time Metal warm work for one ``row_exact_qmv`` launch."""
    probe = mx.zeros((1, k), dtype)
    faces = [_face(lin, probe) for lin in linears]
    if not faces or any(face is None for face in faces):
        return
    if grouped:
        if (
            len({face.vpt for face in faces}) != 1
            or any("bias" in lin for lin in linears)
            or kernels.table_bytes(*faces) > formats.TABLE_BYTES
        ):
            return
    elif len(linears) != 1 or kernels.table_bytes(faces[0]) > formats.TABLE_BYTES:
        return
    names = tuple(formats.tensor_names(lin)[0] for lin in linears)
    operand_keys = tuple((name, *face.tensors) for name, face in zip(names, faces))
    operands = tuple(
        lin[key] for lin, keys in zip(linears, operand_keys) for key in keys
    )
    sizes = tuple(int(lin[name].shape[0]) for lin, name in zip(linears, names))
    resolved = _resolve_geometry(
        linears, faces, sizes, k, rows, geometry, grouped=grouped
    )
    if resolved is None:
        return
    per_group, rps, unrolled = resolved
    plan = row_exact_skeleton.build_plan(
        tuple(faces),
        sizes,
        k,
        rows,
        dtype,
        (per_group, rps, unrolled),
        operands=operands,
        names=names,
        operand_keys=operand_keys,
        grouped=grouped,
    )
    if plan is None:
        return
    if plan.shape is None:
        _register_rolled_instance(plan, k, dtype, instances)
        return
    if plan.shape in unrolled_tasks:
        return

    from ..formats import linear as linear_mod

    for lin in linears:
        if not isinstance(lin, FormatLinear):
            continue
        s_plan = linear_mod._plan(lin)
        if s_plan is None:
            continue
        metal = (dtype, 1) if s_plan.n % 8 else dtype
        if (
            (s_plan.n % 8 and s_plan.shared is None)
            or not s_plan.face.exact_for(dtype)
            or (metal in s_plan.proof.ran and not s_plan.proof.failed)
        ):
            continue
        # Do not skip s_plan.proof.failed so Proof.warm fails closed.
        ref_proof_tasks.setdefault(
            (id(s_plan.proof), metal),
            (
                s_plan.proof,
                metal,
                lambda _stream, sp=s_plan, m=metal: linear_mod._linear_stand_in(sp, m),
            ),
        )

    def _run_unrolled(
        plan=plan,
        linears=linears,
        faces=tuple(faces),
        sizes=sizes,
        k=k,
        rows=rows,
        dtype=dtype,
        per_group=per_group,
        rps=rps,
        operands=operands,
        names=names,
        operand_keys=operand_keys,
        grouped=grouped,
    ) -> None:
        stream = kernels._warm_stream()
        with mx.stream(stream):
            ran_unrolled = False

            def run(p):
                nonlocal ran_unrolled
                ran_unrolled = True
                outputs = plan(p)
                return [_biased(m, y) for m, y in zip(linears, outputs)]

            def reference(p):
                return [_one_row_calls(lin, p) for lin in linears]

            verdict = unrolled_tile_verified(plan.shape, run, reference, rows, k, dtype)
            if verdict:
                in_specs, stand_in = _plan_stand_in(plan, k, dtype)
                inst = kernels.planned(
                    plan.kernel,
                    in_specs,
                    plan.output_dtypes,
                    template=plan.template,
                )
                if ran_unrolled:
                    # plan.kernel already compiled and evaluated on real weights.
                    with inst.kernel._locks.setdefault(inst, threading.Lock()):
                        inst.kernel._built.add(inst)
                else:
                    inst.build(lambda si=stand_in: si)
            else:
                rolled = row_exact_skeleton.build_plan(
                    faces,
                    sizes,
                    k,
                    rows,
                    dtype,
                    (per_group, rps, False),
                    operands=operands,
                    names=names,
                    operand_keys=operand_keys,
                    grouped=grouped,
                )
                if rolled is not None:
                    in_specs, stand_in = _plan_stand_in(rolled, k, dtype)
                    inst = kernels.planned(
                        rolled.kernel,
                        in_specs,
                        rolled.output_dtypes,
                        template=rolled.template,
                    )
                    inst.build(lambda si=stand_in: si)

    unrolled_tasks[plan.shape] = _run_unrolled


def _attn_warm_dtypes(q_proj, model_dtype, *, fp16_recast: bool) -> tuple:
    from ..formats import linear as linear_mod

    if isinstance(q_proj, FormatLinear):
        return linear_mod._layer_warm_dtypes(
            q_proj, model_dtype, None, fp16_recast=fp16_recast
        )
    scales = q_proj.get("scales")
    if isinstance(scales, mx.array) and scales.dtype in linear_mod._WARM_DTYPES:
        return (scales.dtype,)
    if model_dtype is not None:
        return (model_dtype,)
    return linear_mod._WARM_DTYPES


def _row_exact_verify_enabled(model) -> bool:
    """Whether ``model`` opts into row-exact MTP verify (g3 attention path)."""
    for candidate in (
        model,
        getattr(model, "_language_model", None),
        getattr(model, "language_model", None),
    ):
        if getattr(candidate, "_omlx_mtp_row_exact_verify", False):
            return True
    return False


def _gdn_stacked_linears(
    model, *, mtp_enabled: bool = False
) -> tuple[FormatLinear, ...]:
    """Return cached Qwen4 GDN stacked ``FormatLinear`` layers for ``model``."""
    if model is None:
        return ()
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return ()

    from . import qwen35_gdn_prework

    found: list[FormatLinear] = []
    seen: set[int] = set()
    for _, mod in named_modules():
        if not qwen35_gdn_prework._is_qwen4_gdn(mod):
            continue
        dec_plan = qwen35_gdn_prework._qwen4_decode_plan(mod)
        if dec_plan is not None:
            stacked = dec_plan[-1]
            if isinstance(stacked, FormatLinear) and id(stacked) not in seen:
                seen.add(id(stacked))
                found.append(stacked)
        if mtp_enabled:
            ver_plan = qwen35_gdn_prework._qwen4_verify_plan(mod)
            if ver_plan is not None:
                stacked = ver_plan[2]
                if isinstance(stacked, FormatLinear) and id(stacked) not in seen:
                    seen.add(id(stacked))
                    found.append(stacked)
    return tuple(found)


def warm_model(model, *, mtp_enabled: bool = False) -> tuple:
    """Warm served ``omlx_row_exact_qmv`` and GDN stacked ``FormatLinear``
    Metal libraries on ``kernels._WARM_POOL``.

    Always warms the 1-row Qwen4 GDN (stacked in-proj, out-proj), the GDN
    stacked ``FormatLinear`` batched/verify routes (rows 2..MAX_ROWS), and
    fused Attention (group4 Q/K/V/index_qk, group1 o_proj) libraries. When
    ``mtp_enabled`` is True, also warms and verifies fused Attention verify
    (even/odd unrolled group4/group1, live at <2K), plus Attention group3
    (Q/K/V) only for models opting into ``_omlx_mtp_row_exact_verify``.

    Returns futures submitted to ``kernels._WARM_POOL``; callers must await
    them and fail closed on any error.
    """
    if model is None or not mx.metal.is_available():
        return ()
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return ()

    from ..formats import linear as linear_mod
    from . import qwen35_gdn_prework

    fp16_recast = bool(getattr(model, "_omlx_fp16_recast", False))
    model_dtype = None if fp16_recast else linear_mod._infer_model_dtype(model)
    warm_g3 = mtp_enabled and _row_exact_verify_enabled(model)

    instances: dict = {}
    proofs: dict = {}
    unrolled_tasks: dict = {}
    ref_proof_tasks: dict = {}
    seen_attn: set[tuple] = set()
    seen_gdn: set[tuple] = set()

    for _, mod in named_modules():
        # 1. Qwen4 GatedDeltaNet: 1-row row_exact_qmv (in/out) + stacked
        #    FormatLinear Metals (rows >= 2 batch decode and disarmed verify).
        if qwen35_gdn_prework._is_qwen4_gdn(mod):
            projs = tuple(
                getattr(mod, name, None)
                for name in (
                    "in_proj_qkv",
                    "in_proj_z",
                    "in_proj_b",
                    "in_proj_a",
                    "out_proj",
                )
            )
            conv = getattr(mod, "conv1d", None)
            conv_w = getattr(conv, "weight", None)
            if (
                isinstance(conv_w, mx.array)
                and conv_w.dtype in qwen35_gdn_prework._COMPUTE_DTYPES
                and all(isinstance(p, _QUANTIZED_TYPES) for p in projs)
            ):
                dtype = conv_w.dtype
                wide_flag = bool(
                    mod.__dict__.get("_omlx_qwen4_wide_projections", False)
                )
                gdn_key = (
                    dtype,
                    wide_flag,
                    tuple(_linear_sig(p) for p in projs),
                )
                if gdn_key not in seen_gdn:
                    seen_gdn.add(gdn_key)
                    dec_plan = qwen35_gdn_prework._qwen4_decode_plan(mod)
                    ver_plan = (
                        qwen35_gdn_prework._qwen4_verify_plan(mod)
                        if mtp_enabled and dec_plan is None
                        else None
                    )
                    stacked = (
                        dec_plan[-1]
                        if dec_plan is not None
                        else (ver_plan[2] if ver_plan is not None else None)
                    )
                    out_proj = mod.out_proj
                    out_dims = _linear_in_out(out_proj)
                    if dec_plan is not None or ver_plan is not None:
                        if stacked is not None:
                            _plan_warm_target(
                                (stacked,),
                                stacked.input_dims,
                                1,
                                dtype,
                                (1, qwen35_gdn_prework._QWEN4_IN_QMV_RPS, False),
                                grouped=False,
                                instances=instances,
                                unrolled_tasks=unrolled_tasks,
                                ref_proof_tasks=ref_proof_tasks,
                            )
                            if isinstance(stacked, FormatLinear):
                                linear_mod._plan_existing_dense_warm(
                                    stacked, (dtype,), instances, proofs
                                )
                        if out_dims is not None:
                            _plan_warm_target(
                                (out_proj,),
                                out_dims[0],
                                1,
                                dtype,
                                (1, qwen35_gdn_prework._QWEN4_OUT_QMV_RPS, False),
                                grouped=False,
                                instances=instances,
                                unrolled_tasks=unrolled_tasks,
                                ref_proof_tasks=ref_proof_tasks,
                            )
            continue

        # 2. Qwen4 Attention (group4 + group1 always; even/odd _u if MTP;
        #    group3 only for row-exact-verify models).
        q_proj = getattr(mod, "q_proj", None)
        k_proj = getattr(mod, "k_proj", None)
        v_proj = getattr(mod, "v_proj", None)
        o_proj = getattr(mod, "o_proj", None)
        if not (
            isinstance(q_proj, _QUANTIZED_TYPES)
            and isinstance(k_proj, _QUANTIZED_TYPES)
            and isinstance(v_proj, _QUANTIZED_TYPES)
            and isinstance(o_proj, _QUANTIZED_TYPES)
        ):
            continue
        indexer = getattr(mod, "indexer", None)
        idx_proj = getattr(indexer, "index_qk_proj", None)
        has_idx = isinstance(idx_proj, _QUANTIZED_TYPES)
        if not has_idx and not warm_g3:
            continue
        q_dims = _linear_in_out(q_proj)
        o_dims = _linear_in_out(o_proj)
        if q_dims is None or o_dims is None:
            continue
        dtypes = _attn_warm_dtypes(q_proj, model_dtype, fp16_recast=fp16_recast)
        projs = (
            (q_proj, k_proj, v_proj, o_proj, idx_proj)
            if has_idx
            else (
                q_proj,
                k_proj,
                v_proj,
                o_proj,
            )
        )
        attn_key = (dtypes, has_idx, tuple(_linear_sig(p) for p in projs))
        if attn_key in seen_attn:
            continue
        seen_attn.add(attn_key)

        for dtype in dtypes:
            if has_idx:
                g4 = (q_proj, k_proj, v_proj, idx_proj)
                _plan_warm_target(
                    g4,
                    q_dims[0],
                    1,
                    dtype,
                    _fused_attn_tile(1),
                    grouped=True,
                    instances=instances,
                    unrolled_tasks=unrolled_tasks,
                    ref_proof_tasks=ref_proof_tasks,
                )
                _plan_warm_target(
                    (o_proj,),
                    o_dims[0],
                    1,
                    dtype,
                    _fused_attn_tile(1),
                    grouped=True,
                    instances=instances,
                    unrolled_tasks=unrolled_tasks,
                    ref_proof_tasks=ref_proof_tasks,
                )
                if mtp_enabled:
                    for rows in (2, 3):
                        tile = _fused_attn_tile(rows)
                        _plan_warm_target(
                            g4,
                            q_dims[0],
                            rows,
                            dtype,
                            tile,
                            grouped=True,
                            instances=instances,
                            unrolled_tasks=unrolled_tasks,
                            ref_proof_tasks=ref_proof_tasks,
                        )
                        _plan_warm_target(
                            (o_proj,),
                            o_dims[0],
                            rows,
                            dtype,
                            tile,
                            grouped=True,
                            instances=instances,
                            unrolled_tasks=unrolled_tasks,
                            ref_proof_tasks=ref_proof_tasks,
                        )
            if warm_g3:
                g3 = (q_proj, k_proj, v_proj)
                for rows in _VERIFY_WARM_ROWS:
                    _plan_warm_target(
                        g3,
                        q_dims[0],
                        rows,
                        dtype,
                        None,
                        grouped=True,
                        instances=instances,
                        unrolled_tasks=unrolled_tasks,
                        ref_proof_tasks=ref_proof_tasks,
                    )

    # Pre-warm every reference FormatLinear 1-row Metal via Proof.warm before
    # launching unrolled verification workers (zero Proof.prove() races or
    # proof_count() bumps; fail closed and never orphan futures on scan error).
    if ref_proof_tasks:
        ref_futures = kernels.warm_proofs(tuple(ref_proof_tasks.values()))
        first_err: BaseException | None = None
        for fut in ref_futures:
            try:
                fut.result()
            except BaseException as err:  # noqa: BLE001
                if first_err is None:
                    first_err = err
                else:
                    logger.warning(
                        "row_exact_qmv reference proof warm also failed: %s", err
                    )
        if first_err is not None:
            raise first_err

    futures = (
        list(kernels.warm(linear_mod._attach_proofs(instances, proofs)))
        if instances
        else []
    )
    for task in unrolled_tasks.values():
        futures.append(kernels._WARM_POOL.submit(task))
    return tuple(futures)


def warm_model_dispatch(model, *, mtp_enabled: bool = False) -> None:
    """Pre-populate ``row_exact_qmv`` plans and per-thread compiled GDN graphs
    on the engine's inference thread before accepting requests."""
    if model is None or not mx.metal.is_available():
        return
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return

    from ..formats import linear as linear_mod
    from . import qwen35_gdn_prework

    fp16_recast = bool(getattr(model, "_omlx_fp16_recast", False))
    model_dtype = None if fp16_recast else linear_mod._infer_model_dtype(model)
    warm_g3 = mtp_enabled and _row_exact_verify_enabled(model)

    for _, mod in named_modules():
        if qwen35_gdn_prework._is_qwen4_gdn(mod):
            dec_plan = qwen35_gdn_prework._qwen4_decode_plan(mod)
            if dec_plan is not None:
                dtype = mod.conv1d.weight.dtype
                in_qmv, out_qmv = dec_plan[-3], dec_plan[-2]
                if in_qmv is not None:
                    in_qmv(
                        mx.zeros((1, 1, qwen35_gdn_prework._QWEN4_HIDDEN_SIZE), dtype)
                    )
                if out_qmv is not None:
                    out_qmv(mx.zeros((1, 1, mod.num_v_heads * mod.head_v_dim), dtype))
            if mtp_enabled:
                ver_plan = qwen35_gdn_prework._qwen4_verify_plan(mod)
                if ver_plan is not None:
                    dtype = mod.conv1d.weight.dtype
                    in_rows, out_rows = ver_plan[0], ver_plan[1]
                    in_rows(
                        mx.zeros((1, 1, qwen35_gdn_prework._QWEN4_HIDDEN_SIZE), dtype)
                    )
                    out_rows(mx.zeros((1, 1, mod.num_v_heads * mod.head_v_dim), dtype))
            continue

        q_proj = getattr(mod, "q_proj", None)
        k_proj = getattr(mod, "k_proj", None)
        v_proj = getattr(mod, "v_proj", None)
        o_proj = getattr(mod, "o_proj", None)
        if not (
            isinstance(q_proj, _QUANTIZED_TYPES)
            and isinstance(k_proj, _QUANTIZED_TYPES)
            and isinstance(v_proj, _QUANTIZED_TYPES)
            and isinstance(o_proj, _QUANTIZED_TYPES)
        ):
            continue
        q_dims = _linear_in_out(q_proj)
        o_dims = _linear_in_out(o_proj)
        if q_dims is None or o_dims is None:
            continue
        indexer = getattr(mod, "indexer", None)
        idx_proj = getattr(indexer, "index_qk_proj", None)
        has_idx = isinstance(idx_proj, _QUANTIZED_TYPES)
        if not has_idx and not warm_g3:
            continue
        dtypes = _attn_warm_dtypes(q_proj, model_dtype, fp16_recast=fp16_recast)
        for dtype in dtypes:
            x_q = mx.zeros((1, q_dims[0]), dtype)
            x_o = mx.zeros((1, o_dims[0]), dtype)
            if has_idx:
                g4 = (q_proj, k_proj, v_proj, idx_proj)
                _group_plan(g4, x_q, 1, _fused_attn_tile(1))
                _group_plan((o_proj,), x_o, 1, _fused_attn_tile(1))
                if mtp_enabled:
                    for rows in _VERIFY_WARM_ROWS:
                        tile = _fused_attn_tile(rows)
                        _group_plan(g4, x_q, rows, tile)
                        _group_plan((o_proj,), x_o, rows, tile)
            if warm_g3:
                g3 = (q_proj, k_proj, v_proj)
                for rows in _VERIFY_WARM_ROWS:
                    _group_plan(g3, x_q, rows)


__all__ = [
    "MAX_ROWS",
    "RowsQmv",
    "quantized_linear",
    "quantized_linears",
    "quantized_linears_tiled",
    "rows_qmv",
    "warm_model",
    "warm_model_dispatch",
]
