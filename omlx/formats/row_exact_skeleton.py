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

from dataclasses import dataclass
from string import Template
from textwrap import indent

from . import affine

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
        y[(row0 + i) * N_OUT + out_row + r] = static_cast<T>(v);
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
}


@dataclass(frozen=True)
class RowOps:
    """One face's compile-time slots for the row-exact skeleton."""

    helpers: str
    operands: tuple
    slots: dict


def affine_row_ops(bits: int, group_size: int, fast: bool) -> RowOps:
    """The stock affine face's slots, as the per-shape kernels wrote them."""
    return RowOps(
        affine.header(bits, group_size, fast) + affine.decoded(bits),
        ("w", "scales", "biases"),
        _AFFINE_SLOTS,
    )


def dot_row_ops(face) -> RowOps:
    """A Rev-1 plugin or tabled ``Dot`` face's slots.

    The face's ``shared`` Metal declares its struct (``type``) and its
    ``Lanes<K>``; the skeleton binds the face's tensors by name and reads a
    block through the face's own load/decode/apply. The contract check
    (``Lanes`` members, Tables staging) runs once, at construction."""
    helpers = face.source + face.shared
    return RowOps(helpers, tuple(face.tensors), _DOT_SLOTS)


# The generic binding: the face's aggregate (its weight codes, then its
# tensors, then its staged tables) is passed by value; one block's decode and
# apply are the face's Lanes methods.
_DOT_SLOTS = {
    "signature": """template <typename T, int K_SIZE, int N_OUT, int ROWS, int RPS, typename D>""",
    "formals": "    D m,",
    "strides": "  constexpr int BLOCK = D::Lanes<K_SIZE>::BLOCK;",
    "bind": "  auto a = m.lanes(row0);",
    "step": "",
    "aux": "",
    "load": "",
    "fetch": "",
    "apply": "",
    "finish": "0",
}


def emit_row_exact_kernel(ops: RowOps, fast: bool, unrolled: bool, group_count: int, name: str) -> dict:
    """The kernel spec of one row-exact launch (or of ``group_count``
    projections in one launch) as ``mx.fast.metal_kernel`` takes it."""
    block = _BLOCK.substitute(ops.slots)
    last = "" if fast else (
        "  if (int(simd_lid) * VALUES_PER_THREAD < K_SIZE - k) {\n"
        + indent(block, "  ")
        + "  }\n"
    )
    tile = _ROW.substitute(ops.slots).replace("__BLOCK__", block).replace("__LAST_BLOCK__", last)
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
            lines.extend([
                f"    constexpr int TILES_{i} = (N_{i} + 2 * RPS - 1) / (2 * RPS);",
                f"    if (tile < start + TILES_{i}) {{",
                f"      row_exact_tile<T, K_SIZE, N_{i}, ROWS, RPS>(",
                f"          {fields}, tile - start, row0, sg, sl);",
                "      return;",
                "    }",
                f"    start += TILES_{i};",
            ])
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
    return dict(name=name, input_names=inputs, output_names=outputs, header=header, source=body)


def kernel_spec(ops: RowOps, bits: int, group_size: int, fast: bool, unrolled: bool, group_count: int) -> dict:
    """The named spec of one launch, at the affine naming rule."""
    group = f"_group{group_count}" if group_count else ""
    suffix = "_u" if unrolled else ""
    symbol = f"omlx_row_exact_qmv{group}_b{bits}_gs{group_size}_{int(fast)}{suffix}"
    return emit_row_exact_kernel(ops, fast, unrolled, group_count, symbol)
