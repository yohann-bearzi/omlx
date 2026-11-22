# SPDX-License-Identifier: Apache-2.0
"""The row-exact skeleton: one parameterized Metal tile for a row's qmv.

The tile's body is fixed; a face binds its decode/apply arithmetic as
compile-time slots (RowOps), so the stock affine face and every Rev-1 plugin
or tabled face run the SAME skeleton. The affine slots reproduce the
pre-skeleton per-shape kernels byte for byte (specs/parity-skeletons.md §3.1,
240/240 specs). The row and column loops belong to the skeleton; a face
supplies its signature, formals, strides, cursor binding, block step, and
the load/fetch/apply/finish of one block.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import cache
from string import Template
from textwrap import indent
from typing import Any

import mlx.core as mx

from .. import formats
from . import Dot, affine, kernels, tuning
from .tuning import RowIndexWidth

logger = logging.getLogger(__name__)

_ROW = Template(
    r"""
$signature
METAL_FUNC void row_exact_tile(
    const device T* x,
$formals
    device T* y,
    int tile,
    int row0,
    uint simd_gid,
    uint simd_lid) {
  constexpr bool PARTIAL = (N_OUT % (2 * RPS)) != 0;
$strides
  const int out_row = tile * (2 * RPS) + int(simd_gid) * RPS;
  if (PARTIAL && out_row >= N_OUT) {
    return;
  }
  const int valid = PARTIAL ? min(RPS, N_OUT - out_row) : RPS;

$bind
  const device T* xp = x + row0 * K_SIZE + int(simd_lid) * VALUES_PER_THREAD;

  float result[ROWS][RPS];
  ROW_EXACT_UNROLL
  for (int i = 0; i < ROWS; i++) {
    ROW_EXACT_UNROLL
    for (int r = 0; r < RPS; r++) {
      result[i][r] = 0;
    }
  }

  int k = 0;
  const int full_limit = FAST ? K_SIZE : K_SIZE - BLOCK_SIZE;
  for (; k < full_limit; k += BLOCK_SIZE) {
__BLOCK__$step
    xp += BLOCK_SIZE;
  }
__LAST_BLOCK__
  ROW_EXACT_UNROLL
  for (int i = 0; i < ROWS; i++) {
    ROW_EXACT_UNROLL
    for (int r = 0; r < RPS; r++) {
      if (PARTIAL && r >= valid) {
        break;
      }
      float v = $finish;
      if (simd_lid == 0) {
        y[$y_index] = static_cast<T>(v);
      }
    }
  }
}
"""
)

_BLOCK = Template(
    r"""    float xs[ROWS][VALUES_PER_THREAD];
$aux
    ROW_EXACT_UNROLL
    for (int i = 0; i < ROWS; i++) {
      $load
    }
    ROW_EXACT_UNROLL
    for (int r = 0; r < RPS; r++) {
      if (PARTIAL && r >= valid) {
        break;
      }
$fetch
      ROW_EXACT_UNROLL
      for (int i = 0; i < ROWS; i++) {
        $apply
      }
    }
"""
)

# The affine face's slots: the same signature/strides/cursor the per-shape
# kernels carried, so the emitted text is theirs.
_AFFINE_SLOTS = {
    "signature": """// Tiny scale/bias arrays (N < 8) arrive in the constant address space, so
// their pointer type is a template parameter.
template <typename T, int K_SIZE, int N_OUT, int ROWS, int RPS, typename SP>""",
    "formals": """    const device uint32_t* w,
    SP scales,
    SP biases,""",
    "strides": """  const int in_vec_size_w = K_SIZE * BYTES_PER_PACK / PACK_FACTOR;
  const int in_vec_size_g = K_SIZE / GS;""",
    "bind": """  const device uint8_t* ws = (const device uint8_t*)w + out_row * in_vec_size_w +
      int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
  SP sc = scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
  SP bs = biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;""",
    "step": """    ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    sc += BLOCK_SIZE / GS;
    bs += BLOCK_SIZE / GS;""",
    "aux": "    float sums[ROWS];",
    "load": "sums[i] = load_vector(xp + i * K_SIZE, xs[i]);",
    "fetch": """      float wq[W_TERMS];
      decode_w(ws + r * in_vec_size_w, wq);
      float s = sc[r * in_vec_size_g];
      float b = bs[r * in_vec_size_g];""",
    "apply": "result[i][r] += s * wdot(wq, xs[i]) + sums[i] * b;",
    "finish": "simd_sum(result[i][r])",
    "y_index": "(row0 + i) * N_OUT + out_row + r",
}

# 64-bit row/group offsets selected at construction when N * row_bytes >= 2**31.
_AFFINE_SLOTS_64 = {
    **_AFFINE_SLOTS,
    "strides": (
        "  const size_t in_vec_size_w ="
        " size_t(K_SIZE) * BYTES_PER_PACK / PACK_FACTOR;\n"
        "  const size_t in_vec_size_g = size_t(K_SIZE) / GS;"
    ),
    "bind": (
        "  const device uint8_t* ws = (const device uint8_t*)w +"
        " size_t(out_row) * in_vec_size_w +\n"
        "      int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;\n"
        "  SP sc = scales + size_t(out_row) * in_vec_size_g +"
        " int(simd_lid) / SCALE_STEP_PER_THREAD;\n"
        "  SP bs = biases + size_t(out_row) * in_vec_size_g +"
        " int(simd_lid) / SCALE_STEP_PER_THREAD;"
    ),
    "y_index": "size_t(row0 + i) * N_OUT + size_t(out_row + r)",
}

# Generic Rev-1 Dot face slots: the face's aggregate (packed weight codes, its
# tensors, then its staged tables) is passed by value as D; one block's load,
# decode, apply, step and finish are the face's Lanes<K_SIZE> methods.
_DOT_SLOTS = {
    "signature": (
        "template <typename T, int K_SIZE, int N_OUT, int ROWS, int RPS, typename D>"
    ),
    "formals": "    D m,",
    "strides": (
        "  using L = typename D::template Lanes<K_SIZE>;\n"
        "  constexpr int VALUES_PER_THREAD = L::VPT;\n"
        "  constexpr int BLOCK_SIZE = L::BLOCK;\n"
        "  constexpr bool FAST = (K_SIZE % BLOCK_SIZE) == 0;"
    ),
    "bind": "  L a(m, size_t(out_row), simd_lid);",
    "step": "    a.step();",
    "aux": "    decltype(L::load(xp, xs[0])) aux[ROWS];",
    "load": "aux[i] = L::load(xp + i * K_SIZE, xs[i]);",
    "fetch": "      const auto d = L::decode(a.row(r));",
    "apply": (
        "thread const float* xt = xs[i];\n"
        "        result[i][r] = L::apply(result[i][r], d, xt, aux[i]);"
    ),
    "finish": "a.finish(r, metal::simd_sum(result[i][r]))",
    "y_index": "size_t(row0 + i) * N_OUT + size_t(out_row + r)",
}

_DOT_SLOTS_FAST = {
    **_DOT_SLOTS,
    "strides": (
        _DOT_SLOTS["strides"]
        + '\n  static_assert(FAST, "row-exact fast kernel: K must be whole blocks");'
    ),
}

# Cold-proof contract check: verifies a Rev-1 Dot face's contract once on a
# minimal zero stand-in so runtime kernels omit the SFINAE / static_assert
# boilerplate (specs/parity-skeletons.md §2.4).
_DOT_CONTRACT_CHECK = kernels._compact(
    r"""
template <typename T, int K_SIZE, int N_OUT, int VPT, typename D, typename XP>
METAL_FUNC void omlx_row_exact_contract(
    XP x, D m, device T* y, uint sg, uint sl) {
  const auto xc = omlx_inputs<T>(x);
  constexpr int FACE = D::CONTRACT;
  (void)sizeof(omlx_revision<FACE>);
  if constexpr (FACE <= OMLX_CONTRACT && FACE >= OMLX_OLDEST_CONTRACT) {
    using L = typename D::template Lanes<K_SIZE>;
    static_assert(
        L::VPT >= 1,
        "dot face: Lanes<K>::VPT must be one input or more");
    if constexpr (L::VPT >= 1) {
      static_assert(
          L::BLOCK == 32 * L::VPT,
          "dot face: Lanes<K>::BLOCK must be VPT inputs for each of a simdgroup's 32 "
          "lanes");
      static_assert(
          K_SIZE % L::VPT == 0,
          "dot face: K must be a whole number of a lane's VPT inputs");
      static_assert(
          L::VPT == VPT,
          "dot face: Lanes<K>::VPT must be the face's vpt, which sizes the launch");
      row_exact_tile<T, K_SIZE, N_OUT, 1, 1>(xc, m, y, 0, 0, sg, sl);
    }
  }
}
"""
)


@dataclass(frozen=True)
class RowOps:
    """One face's compile-time slots for the row-exact skeleton."""

    helpers: str
    operands: tuple
    slots: dict
    face: Dot | None = None


def affine_row_ops(
    bits: int,
    group_size: int,
    fast: bool,
    index_width: RowIndexWidth = RowIndexWidth.INT32_CHECKED,
) -> RowOps:
    """The stock affine face's slots, as the per-shape kernels wrote them."""
    slots = (
        _AFFINE_SLOTS_64
        if index_width is RowIndexWidth.SIZE_T64
        else _AFFINE_SLOTS
    )
    return RowOps(
        affine.header(bits, group_size, fast) + affine.decoded(bits),
        ("w", "scales", "biases"),
        slots,
    )


def dot_row_ops(face: Dot) -> RowOps:
    """A Rev-1 plugin or tabled ``Dot`` face's slots for ``RowExactSkeleton``.

    The face's ``source`` (and optional ``shared``) Metal declares its struct
    (``type``) and its ``Lanes<K>``; the skeleton binds the face's tensors and
    stages its ``tables`` through ``kernels.Tables``, then reads every block
    through the face's own load/decode/apply/finish. The contract check runs
    once at cold proof (``prove_dot_face``), so runtime kernels stay slim."""
    if not isinstance(face, Dot):
        raise TypeError(f"dot_row_ops: expected a Dot face, got {type(face).__name__}")
    if face.vpt < 1:
        raise ValueError(
            f"dot_row_ops: face {face.key!r} states no vpt, which sizes a "
            "row-exact launch"
        )
    return RowOps(
        face.emitted_source + face.emitted_shared,
        ("weight", *face.tensors),
        _DOT_SLOTS,
        face,
    )


def _render_tile(slots: dict, fast: bool) -> str:
    block = _BLOCK.substitute(slots)
    last = (
        ""
        if fast
        else (
            "  if (int(simd_lid) * VALUES_PER_THREAD < K_SIZE - k) {\n"
            + indent(block, "  ")
            + "  }\n"
        )
    )
    return (
        _ROW.substitute(slots)
        .replace("__BLOCK__", block)
        .replace("__LAST_BLOCK__", last)
    )


def _dedup_face_sources(faces: tuple[Dot, ...], *, named: bool) -> str:
    sources: dict[str, str] = {}
    for phase in ("main", "shared"):
        for face in faces:
            for part_name, when, source in face._metal_parts():
                if when == phase and (source or part_name == "source"):
                    key = (
                        face.key
                        if part_name == "source"
                        else f"{face.key} {part_name}"
                    )
                    sources.setdefault(source, key)
    if named:
        return "".join(
            kernels._named(f"<quant format {key}>", src) for src, key in sources.items()
        )
    return "".join(f"{src}\n" for src in sources)


def _emit_dot_kernel(
    ops_seq: tuple[RowOps, ...],
    fast: bool,
    unrolled: bool,
    group_count: int,
    name: str,
) -> dict:
    faces = tuple(op.face for op in ops_seq)
    tables = kernels.Tables()
    inputs = ["x"]
    decls = []
    for i, face in enumerate(faces):
        staged = tuple(tables.add(t) for t in face.tables)
        prefix = f"m{i}" if group_count else "m"
        args = [f"{prefix}_weight", *(f"{prefix}_{t}" for t in face.tensors)]
        inputs.extend(args)
        members = ", ".join([f"(const device uint8_t*){args[0]}", *args[1:], *staged])
        decls.append(f"    const auto {prefix} = {face.type}{{{members}}};\n")

    slots = _DOT_SLOTS_FAST if fast else _DOT_SLOTS
    tile = _render_tile(slots, fast)
    pragma = '_Pragma("clang loop unroll(full)")' if unrolled else ""
    header = (
        f"#define ROW_EXACT_UNROLL {pragma}\n"
        + kernels._INDEX_PACKS
        + tables.header
        + _dedup_face_sources(faces, named=False)
        + tile
    )

    if group_count:
        lines = [
            "    const int tile = int(threadgroup_position_in_grid.z);",
            "    const int row0 = int(threadgroup_position_in_grid.y) * ROWS;",
            "    const uint sg = simdgroup_index_in_threadgroup;",
            "    const uint sl = thread_index_in_simdgroup;",
            "    int start = 0;",
        ]
        outputs = []
        for i in range(group_count):
            outputs.append(f"y{i}")
            lines.extend(
                [
                    f"    constexpr int TILES_{i} = (N_{i} + 2 * RPS - 1) / (2 * RPS);",
                    f"    if (tile < start + TILES_{i}) {{",
                    f"      row_exact_tile<T, K_SIZE, N_{i}, ROWS, RPS>(",
                    f"          x, m{i}, y{i}, tile - start, row0, sg, sl);",
                    "      return;",
                    "    }",
                    f"    start += TILES_{i};",
                ]
            )
        dispatch = "\n".join(lines) + "\n"
    else:
        outputs = ["y"]
        dispatch = (
            "\n    row_exact_tile<T, K_SIZE, N_SIZE, ROWS, RPS>(\n"
            "        x, m, y,\n"
            "        int(threadgroup_position_in_grid.z),\n"
            "        int(threadgroup_position_in_grid.y) * ROWS,\n"
            "        simdgroup_index_in_threadgroup,\n"
            "        thread_index_in_simdgroup);\n"
        )
    body = tables.prologue + "".join(decls) + dispatch
    return dict(
        name=name,
        input_names=inputs,
        output_names=outputs,
        header=header,
        source=body,
    )


def emit_row_exact_kernel(
    ops: RowOps | tuple[RowOps, ...] | list[RowOps],
    fast: bool,
    unrolled: bool,
    group_count: int,
    name: str,
) -> dict:
    """The kernel spec of one row-exact launch (or of ``group_count``
    projections in one launch) as ``mx.fast.metal_kernel`` takes it."""
    if isinstance(ops, (tuple, list)):
        ops_seq = tuple(ops)
        if not ops_seq or any(
            not isinstance(op, RowOps) or op.face is None for op in ops_seq
        ):
            raise TypeError(
                "emit_row_exact_kernel: a sequence of RowOps must hold dot_row_ops"
            )
        expected = group_count or 1
        if len(ops_seq) != expected:
            raise ValueError(
                f"emit_row_exact_kernel: got {len(ops_seq)} ops "
                f"for group_count={group_count}"
            )
        return _emit_dot_kernel(ops_seq, fast, unrolled, group_count, name)
    if ops.face is not None:
        return _emit_dot_kernel(
            (ops,) * (group_count or 1), fast, unrolled, group_count, name
        )

    tile = _render_tile(ops.slots, fast)
    pragma = '_Pragma("clang loop unroll(full)")' if unrolled else ""
    header = f"#define ROW_EXACT_UNROLL {pragma}\n" + ops.helpers + tile
    inputs = ["x"]
    if group_count:
        lines = [
            "    const int tile = int(threadgroup_position_in_grid.z);",
            "    const int row0 = int(threadgroup_position_in_grid.y) * ROWS;",
            "    const uint sg = simdgroup_index_in_threadgroup;",
            "    const uint sl = thread_index_in_simdgroup;",
            "    int start = 0;",
        ]
        outputs = []
        for i in range(group_count):
            args = [f"{operand}{i}" for operand in ops.operands]
            inputs.extend(args)
            outputs.append(f"y{i}")
            fields = ", ".join(["x", *args, f"y{i}"])
            lines.extend(
                [
                    f"    constexpr int TILES_{i} = (N_{i} + 2 * RPS - 1) / (2 * RPS);",
                    f"    if (tile < start + TILES_{i}) {{",
                    f"      row_exact_tile<T, K_SIZE, N_{i}, ROWS, RPS>(",
                    f"          {fields}, tile - start, row0, sg, sl);",
                    "      return;",
                    "    }",
                    f"    start += TILES_{i};",
                ]
            )
        body = "\n".join(lines) + "\n"
    else:
        inputs.extend(ops.operands)
        outputs = ["y"]
        fields = ", ".join([*inputs, *outputs])
        body = (
            "\n    row_exact_tile<T, K_SIZE, N_SIZE, ROWS, RPS>(\n"
            f"        {fields},\n"
            "        int(threadgroup_position_in_grid.z),\n"
            "        int(threadgroup_position_in_grid.y) * ROWS,\n"
            "        simdgroup_index_in_threadgroup,\n"
            "        thread_index_in_simdgroup);\n"
        )
    return dict(
        name=name,
        input_names=inputs,
        output_names=outputs,
        header=header,
        source=body,
    )


def kernel_spec(
    ops: RowOps,
    bits: int,
    group_size: int,
    fast: bool,
    unrolled: bool,
    group_count: int,
    index_width: RowIndexWidth = RowIndexWidth.INT32_CHECKED,
) -> dict:
    """The named spec of one launch, at the affine naming rule."""
    group = f"_group{group_count}" if group_count else ""
    suffix = "_u" if unrolled else ""
    wide = "_i64" if index_width is RowIndexWidth.SIZE_T64 else ""
    symbol = (
        f"omlx_row_exact_qmv{group}_b{bits}_gs{group_size}_{int(fast)}{suffix}{wide}"
    )
    return emit_row_exact_kernel(ops, fast, unrolled, group_count, symbol)


@cache
def _dot_contract_kernel(face: Dot):
    tables = kernels.Tables()
    staged = tuple(tables.add(t) for t in face.tables)
    args = ["m_weight", *(f"m_{t}" for t in face.tensors)]
    members = ", ".join([f"(const device uint8_t*){args[0]}", *args[1:], *staged])
    tile = _render_tile(_DOT_SLOTS, fast=False)
    header = (
        kernels._named(kernels._SKELETONS, kernels._INDEX_PACKS)
        + tables.header
        + _dedup_face_sources((face,), named=True)
        + "\n#define ROW_EXACT_UNROLL \n"
        + tile
        + kernels._named(
            kernels._SKELETONS,
            kernels._revisions() + kernels._CHECKS + _DOT_CONTRACT_CHECK,
            kernels._REVISIONS_LINE,
        )
        + kernels._REVISIONS_END
        + kernels._named("<kernel header>", "")
    )
    body = (
        tables.prologue
        + kernels._members_check(face)
        + f"    const auto m = {face.type}{{{members}}};\n"
        + kernels._named("<kernel>", "")
        + "    omlx_row_exact_contract<T, K_SIZE, N_OUT, VPT>(\n"
        + "        x, m, y,\n"
        + "        simdgroup_index_in_threadgroup,\n"
        + "        thread_index_in_simdgroup);\n"
    )
    return mx.fast.metal_kernel(
        name=f"omlx_row_exact_contract_{face.key}",
        input_names=["x", *args],
        output_names=["y"],
        header=header,
        source=body,
    )


def prove_dot_face(
    face: Dot,
    k: int,
    n: int,
    dtype: mx.Dtype,
    operand_specs: tuple[tuple[tuple[int, ...], mx.Dtype], ...],
) -> bool:
    """Compile-check a Rev-1 ``Dot`` face once on a minimal zero stand-in.

    Validates ``CONTRACT``, ``VPT``, ``BLOCK == 32 * VPT``, ``K % VPT == 0``,
    ``Lanes<K>::VPT == face.vpt``, declared ``omlx_table`` member types, and
    instantiation through ``row_exact_tile`` via ``kernels.Proof``. Returns
    False (and caches the failure) if the face fails the contract."""
    if not isinstance(face, Dot) or face.vpt < 1 or k < 8 or n < 1:
        return False
    if len(operand_specs) != 1 + len(face.tensors):
        return False
    kinds = tuple(
        (dt, kernels._input_kind(len(shape) == 0, math.prod(shape)))
        for shape, dt in operand_specs
    )
    proof = kernels.proof("row_exact_dot_contract", face, k, kinds)
    stand_in_n = min(n, 2)
    metal = (dtype, stand_in_n)

    def build():
        kern = _dot_contract_kernel(face)
        inputs = kernels.zeros(((1, k), dtype), *operand_specs)

        def launch(*arrays):
            return kern(
                inputs=list(arrays),
                template=[
                    ("T", dtype),
                    ("K_SIZE", k),
                    ("N_OUT", stand_in_n),
                    ("VPT", face.vpt),
                ],
                grid=(32, 2, 1),
                threadgroup=(32, 2, 1),
                output_shapes=[(1, stand_in_n)],
                output_dtypes=[dtype],
            )[0]

        return inputs, launch

    return proof.prove(
        metal,
        build,
        logger,
        f"quant format face {face.key}: row-exact skeleton engaged",
        f"quant format face {face.key}: row-exact skeleton contract failed; fallback",
    )


@cache
def _affine_spec(
    bits: int,
    group_size: int,
    fast: bool,
    unrolled: bool,
    group_count: int,
    index_width: RowIndexWidth,
) -> dict:
    ops = affine_row_ops(bits, group_size, fast, index_width)
    return kernel_spec(ops, bits, group_size, fast, unrolled, group_count, index_width)


@cache
def _dot_spec(
    faces: tuple[Dot, ...], fast: bool, unrolled: bool, group_count: int
) -> dict:
    ops = tuple(dot_row_ops(face) for face in faces)
    group = f"_group{group_count}" if group_count else ""
    suffix = "_u" if unrolled else ""
    keys = "_".join(face.key for face in faces)
    symbol = f"omlx_row_exact_dot{group}_{keys}_{int(fast)}{suffix}"
    return emit_row_exact_kernel(ops, fast, unrolled, group_count, symbol)


@cache
def _kernel_from_spec(
    name: str,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    header: str,
    source: str,
):
    return mx.fast.metal_kernel(
        name=name,
        input_names=list(input_names),
        output_names=list(output_names),
        header=header,
        source=source,
    )


def compile_kernel(spec: dict):
    """Compile one emitted row-exact kernel spec into an MLX Metal kernel.

    The unroll choice is part of the spec (its name and header), so a caller
    cannot compile another variant through this entry point."""
    return _kernel_from_spec(
        spec["name"],
        tuple(spec["input_names"]),
        tuple(spec["output_names"]),
        spec["header"],
        spec["source"],
    )


@dataclass(frozen=True, slots=True)
class RowExactPlan:
    """Immutable launch plan built once at construction / first signature.

    Holds every construction-time decision (rule 28): compiled kernel, spec,
    grid, threadgroup, template, output shapes/dtypes, index width,
    pre-flattened operands and identity checks, and optional
    unrolled-verification key. Hot ``__call__`` performs zero geometry,
    source, or operand assembly.
    """

    kernel: Any
    spec: dict
    grid: tuple[int, int, int]
    threadgroup: tuple[int, int, int]
    template: list[tuple[str, Any]]
    output_shapes: list[tuple[int, ...]]
    output_dtypes: list[mx.Dtype]
    padding: int
    faces: tuple[Dot, ...]
    names: tuple[str, ...]
    index_width: RowIndexWidth
    operands: tuple[mx.array, ...]
    operand_keys: tuple[tuple[str, ...], ...]
    shape: tuple | None
    direct_input: bool = False
    single_checks: tuple[tuple[str, mx.array], ...] = ()
    group_checks: tuple[tuple[int, str, mx.array], ...] = ()

    def operands_match(self, linears) -> bool:
        """True when every layer's tensors are still the exact arrays cached
        in ``self.operands`` (lock-free, zero ``id()`` audit events)."""
        dget = dict.get
        for idx, key, arr in self.group_checks:
            if dget(linears[idx], key) is not arr:
                return False
        return True

    def __call__(
        self, x: mx.array, operands: tuple[mx.array, ...] | None = None
    ) -> list:
        """Launch the planned kernel for ``x`` using pre-flattened operands
        (or explicit ``operands`` when a compiled graph passes traced arrays)."""
        ops = self.operands if operands is None else operands
        if not self.direct_input:
            if not self.padding:
                return self.kernel(
                    inputs=[x, *ops],
                    template=self.template,
                    grid=self.grid,
                    threadgroup=self.threadgroup,
                    output_shapes=self.output_shapes,
                    output_dtypes=self.output_dtypes,
                )
            rows = x.shape[0]
            outputs = self.kernel(
                inputs=[mx.pad(x, [(0, self.padding), (0, 0)]), *ops],
                template=self.template,
                grid=self.grid,
                threadgroup=self.threadgroup,
                output_shapes=self.output_shapes,
                output_dtypes=self.output_dtypes,
            )
            return [y[:rows] for y in outputs]
        return self.kernel(
            inputs=[x, *ops],
            template=self.template,
            grid=self.grid,
            threadgroup=self.threadgroup,
            output_shapes=[(*x.shape[:-1], self.output_shapes[0][-1])],
            output_dtypes=self.output_dtypes,
        )


def build_plan(
    faces: tuple[Dot, ...],
    shapes: tuple[int, ...],
    k: int,
    rows: int,
    dtype: mx.Dtype,
    geometry: tuple[int, int, bool],
    *,
    operands: tuple[mx.array, ...] = (),
    names: tuple[str, ...] = (),
    operand_keys: tuple[tuple[str, ...], ...] = (),
    grouped: bool = False,
    index_width: RowIndexWidth | None = None,
    direct_input: bool = False,
    row_bytes: int | None = None,
) -> RowExactPlan | None:
    """Build an immutable ``RowExactPlan`` for one or more ``Dot`` faces.

    Resolves affine-slot vs generic ``dot_row_ops`` binding, ``RowIndexWidth``,
    cold-proof contract verification for plugin/tabled/mixed faces, kernel
    compilation, template arguments, grid, and pre-flattened operands once at
    construction time (rules 25, 28).
    """
    faces = tuple(faces)
    shapes = tuple(int(n) for n in shapes)
    per_group, rps, unrolled = geometry
    if (
        not faces
        or len(faces) != len(shapes)
        or k < 8
        or rows < 1
        or per_group < 1
        or rps < 1
        or any(n < 1 for n in shapes)
    ):
        return None
    if len({face.vpt for face in faces}) != 1 or faces[0].vpt < 1:
        return None
    if not grouped and len(faces) != 1:
        return None

    padding = -rows % per_group
    padded = rows + padding
    # A direct-input plan launches ``x`` as it is and shapes one output: only
    # an unpadded single-projection launch can take it.
    if direct_input and (grouped or padding):
        return None
    affine_specs = [affine.row_ops_spec(face) for face in faces]
    homogeneous_affine = (
        affine_specs[0] is not None
        and all(spec == affine_specs[0] for spec in affine_specs)
    )

    max_n = max(shapes)
    if index_width is None:
        if row_bytes is not None:
            rb = row_bytes
        elif homogeneous_affine:
            rb = k * affine_specs[0][0] // 8
        elif operands:
            rb = max(
                (a.nbytes // max(a.shape[0], 1) for a in operands if a.ndim >= 1),
                default=k,
            )
        else:
            rb = k
        index_width = tuning.row_index_width(max_n, max(rb, padded))

    group_count = len(faces) if grouped else 0
    if homogeneous_affine:
        bits, group_size, fast = affine_specs[0]
        spec = _affine_spec(bits, group_size, fast, unrolled, group_count, index_width)
        kernel = compile_kernel(spec)
        shape_key = (
            (
                "affine",
                bits,
                group_size,
                fast,
                index_width,
                group_count,
                dtype,
                k,
                shapes,
                per_group,
                rps,
            )
            if unrolled
            else None
        )
    else:
        if kernels.table_bytes(*faces) > formats.TABLE_BYTES:
            return None
        if operands:
            pos = 0
            proven: set[tuple] = set()
            for face, n in zip(faces, shapes):
                arity = 1 + len(face.tensors)
                if pos + arity > len(operands):
                    return None
                op_specs = tuple(
                    (a.shape, a.dtype) for a in operands[pos : pos + arity]
                )
                pos += arity
                key = (face, n, op_specs)
                if key not in proven:
                    if not prove_dot_face(face, k, n, dtype, op_specs):
                        return None
                    proven.add(key)
        vpt = faces[0].vpt
        fast = (k % (32 * vpt)) == 0
        spec = _dot_spec(faces, fast, unrolled, group_count)
        kernel = compile_kernel(spec)
        shape_key = (
            ("dot", faces, fast, group_count, dtype, k, shapes, per_group, rps)
            if unrolled
            else None
        )

    if group_count:
        template = [
            ("T", dtype),
            ("K_SIZE", k),
            *((f"N_{i}", n) for i, n in enumerate(shapes)),
            ("ROWS", per_group),
            ("RPS", rps),
        ]
    else:
        template = [
            ("T", dtype),
            ("K_SIZE", k),
            ("N_SIZE", shapes[0]),
            ("ROWS", per_group),
            ("RPS", rps),
        ]
    tiles = sum((n + 2 * rps - 1) // (2 * rps) for n in shapes)
    grid = (32, 2 * (padded // per_group), tiles)
    threadgroup = (32, 2, 1)
    output_shapes = [(padded, n) for n in shapes]
    output_dtypes = [dtype] * len(shapes)
    operands_t = tuple(operands)
    keys_t = tuple(operand_keys)
    group_checks: list[tuple[int, str, mx.array]] = []
    if operands_t and keys_t:
        pos = 0
        for idx, keys in enumerate(keys_t):
            for key in keys:
                group_checks.append((idx, key, operands_t[pos]))
                pos += 1
    single_checks = (
        tuple((key, arr) for _, key, arr in group_checks)
        if len(keys_t) == 1
        else ()
    )
    return RowExactPlan(
        kernel=kernel,
        spec=spec,
        grid=grid,
        threadgroup=threadgroup,
        template=template,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        padding=padding,
        faces=faces,
        names=tuple(names),
        index_width=index_width,
        operands=operands_t,
        operand_keys=keys_t,
        shape=shape_key,
        direct_input=direct_input,
        single_checks=single_checks,
        group_checks=tuple(group_checks),
    )
