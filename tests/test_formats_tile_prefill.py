# SPDX-License-Identifier: Apache-2.0
"""Sorted prefill through a format's tile face on NAX and steel GPUs."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from omlx import formats
from omlx.formats import Tile, kernels, testing
from omlx.formats.linear import FormatSwitchLinear
from omlx.patches import m5_gather_qmm_nax as nax
from omlx.patches import moe_expert_offload
from tests.test_formats_kernels import _toy as _dot_toy


@pytest.fixture(autouse=True)
def by_hand():
    with getattr(testing, "verified", nullcontext)("qwen4_exp"):
        yield


def _on_nax() -> bool:
    try:
        from omlx.custom_kernels.nax import is_nax_available

        return bool(is_nax_available())
    except Exception:  # noqa: BLE001
        return False


_NAX = _on_nax()
_TILE_PROOF = "format_switch_tile_nax" if _NAX else "format_switch_tile_steel"
needs_nax = pytest.mark.skipif(not _NAX, reason="needs an M5 (NAX) GPU")
needs_steel = pytest.mark.skipif(_NAX, reason="needs the steel sorted gather")

# A format of 8-bit codes around 128 and a float32 scale per row (the dot
# face of tests/test_formats_kernels.py), with a tile face of the same
# matrix: w = (code - 128) * scale.
_TILE = r"""namespace omlx_tiletoy_KEY {
struct Tile {
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* codes;
  const device float* scales;

  template <int K>
  struct Runs {
    static constant constexpr int RUN = 8;
    static constant constexpr int GROUP = 8;
    struct P {
      float s;
    };
    using Codes = uint2;

    const device packed_uint2* c;
    const device float* r;

    Runs(Tile m, size_t row)
        : c((const device packed_uint2*)(m.codes + row * K)), r(m.scales + row) {}

    P params(int) const {
      return {r[0]};
    }
    Codes fetch(int run) const {
      return Codes(c[run]);
    }
    __OMLX_READ_WEIGHT__ {
      const uint code = ((i < 4 ? q.x : q.y) >> (8 * (i & 3))) & 0xFFu;
      return (float(code) - 128.0f) * p.s;
    }
  };
};
}  // namespace omlx_tiletoy_KEY
"""


_TILE_READS = formats.TileReads(weight=formats.WeightRead(params=("p", "q", "i")))


def _tile_toy(key, source=_TILE, reads=_TILE_READS) -> Tile:
    return Tile(
        key=f"tiletoy_{key}",
        source=source.replace("KEY", key),
        type=f"omlx_tiletoy_{key}::Tile",
        tensors=("group_scales",),
        reads=reads,
    )


class _ToyFormat(formats.QuantFormat):
    mode, codes_name, tensors, contract = "tiletoy", "codes", ("group_scales",), (
        formats.CONTRACT
    )

    def __init__(self, dot, tile):
        self.dot_face, self.tile_face = dot, tile

    def layout(self, k, n, bits, group_size):
        if (bits, group_size) != (8, 1):
            return None
        scales = {name: ((n,), mx.float32) for name in self.tensors}
        return {self.codes_name: ((n, k), mx.uint8), **scales}

    def dot(self, layer, k, n):
        return self.dot_face

    def tile(self, layer, k, n):
        return self.tile_face

    def decode(self, tensors, bits, group_size):
        codes = tensors["codes"].astype(mx.float32) - 128.0
        return codes * mx.expand_dims(tensors["group_scales"], -1)


@pytest.fixture
def toy(monkeypatch):
    """Registers the toy format with a dot face and the tile face given."""
    formats.load()

    def register(tile):
        fmt = _ToyFormat(_dot_toy("prefill"), tile)
        monkeypatch.setitem(formats._FORMATS, "tiletoy", fmt)
        return fmt

    return register


# Rows an expert: an empty expert, runs spanning tiles, partial tiles.
_COUNTS = (70, 0, 5, 33, 64, 17, 1, 130, 0, 11)
E = len(_COUNTS)


def _stack(n, k, seed=0):
    mx.random.seed(seed)
    codes = mx.random.randint(0, 256, (E, n, k)).astype(mx.uint8)
    scales = mx.random.normal((E, n)) * 0.01
    return {"codes": codes, "group_scales": scales}


def _rows(k, counts=_COUNTS, dtype=mx.bfloat16, seed=1):
    idx = mx.array(np.repeat(np.arange(len(counts)), counts).astype(np.uint32))
    mx.random.seed(seed)
    x = (mx.random.normal((int(idx.shape[0]), 1, k)) * 0.5).astype(dtype)
    mx.eval(x, idx)
    return x, idx


def _want(tensors, x, idx):
    """MLX's sorted gather over the decode rounded once to x's type: the
    NAX route's bits (TILE-DESIGN.md 4.3, TM8)."""
    fmt = formats.get("tiletoy")
    w = fmt.decode(tensors, 8, 1).astype(x.dtype)
    return mx.gather_mm(x, w.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=True)


def _same(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and mx.array_equal(a, b).item()


def test_a_sorted_shape_the_tile_refuses_decodes_bounded_pieces(toy, monkeypatch):
    from omlx.formats import linear

    k = n = 64
    fmt = toy(_tile_toy("bounded_refusal"))
    fmt.dot_face = replace(fmt.dot_face, dtypes=(mx.float16,))
    tensors = _stack(n, k)
    layer = FormatSwitchLinear("tiletoy", 8, 1, k, tensors)
    indices = mx.repeat(mx.arange(E, dtype=mx.uint32), 4)
    x = mx.broadcast_to(mx.eye(k, dtype=mx.bfloat16)[:2], (indices.size, 2, k))
    want = _want(tensors, x, indices)
    widths = []
    decode = fmt.decode

    def decoded(arrays, bits, group_size):
        widths.append(arrays["codes"].shape[0])
        return decode(arrays, bits, group_size)

    monkeypatch.setattr(fmt, "decode", decoded)
    # Warm the one-row-per-index shape. Its graph cannot serve two rows.
    mx.eval(layer(x[:, :1], indices, sorted_indices=True))
    if _on_nax():
        assert widths == []
    widths.clear()

    # The rows face admits float16 only. A piece holds three matrices.
    monkeypatch.setattr(linear, "_DECODE_BYTES", 3 * n * k * (4 + x.dtype.size))
    hints = []
    gather = mx.gather_mm

    def gathered(*args, **kwargs):
        hints.append(kwargs["sorted_indices"])
        return gather(*args, **kwargs)

    monkeypatch.setattr(mx, "gather_mm", gathered)
    assert _same(layer(x, indices, sorted_indices=True), want)
    assert widths == [3, 3, 3, 1]
    assert hints == [True] * 4


def _proof(name):
    (proof,) = [p for key, p in kernels._PROOFS.items() if key[0] == name]
    return proof


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("k, n", [(256, 128), (320, 96), (640, 256)])
def test_a_format_stack_s_sorted_rows_take_the_tile_route(toy, caplog, k, n, dtype):
    toy(_tile_toy("route"))
    tensors = _stack(n, k, seed=k + n)
    layer = FormatSwitchLinear("tiletoy", 8, 1, k, tensors)
    x, idx = _rows(k, dtype=dtype)
    with caplog.at_level(logging.INFO, logger="omlx.formats.linear"):
        got = layer(x, idx, sorted_indices=True)
        mx.eval(got)
    assert _same(got, _want(tensors, x, idx))
    tile = _proof(_TILE_PROOF)
    assert tile.ran and not tile.failed and not _proof("format_switch_linear").ran
    assert "quant format face tiletoy_route: tile kernel engaged" in caplog.text
    # The second call replays the kept graph.
    (kept,) = tile.graphs.values()
    assert _same(layer(x, idx, sorted_indices=True), got)
    assert list(tile.graphs.values()) == [kept]


@pytest.mark.parametrize("case", ["control", "const_face", "int_weight_index"])
def test_sorted_tile_calls_the_checked_face_overloads(toy, case):
    source = _TILE
    if case == "const_face":
        source = source.replace(
            "Runs(Tile m, size_t row)",
            "Runs(thread Tile&, size_t) = delete;\n"
            "    Runs(thread const Tile& m, size_t row)",
        )
    if case == "int_weight_index":
        source = source.replace(
            "    __OMLX_READ_WEIGHT__ {",
            "    static float weight(thread const P&, Codes, short) { return 0; }\n"
            "    __OMLX_READ_WEIGHT__ {",
        )
    toy(_tile_toy("checked_" + case, source))
    tensors = _stack(128, 256, seed=122)
    layer = FormatSwitchLinear("tiletoy", 8, 1, 256, tensors)
    x, ids = _rows(256)
    got = layer(x, ids, sorted_indices=True)
    assert _same(got, _want(tensors, x, ids))
    tile = _proof(_TILE_PROOF)
    assert tile.ran and not tile.failed


@pytest.mark.parametrize("bits, group_size", [(2, 64), (3, 64), (4, 64), (8, 32)])
def test_affine_through_a_format_stack_is_mlx_s_sorted_gather_qmm(bits, group_size):
    k, n = 512, 128
    mx.random.seed(3)
    w = (mx.random.normal((E, n, k)) * 0.05).astype(mx.bfloat16)
    wq, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    tensors = {"weight": wq, "scales": scales, "biases": biases}
    layer = FormatSwitchLinear("affine", bits, group_size, k, tensors)
    x, idx = _rows(k)
    got = layer(x, idx, sorted_indices=True)
    want = nax._stock_gather_qmm()(
        x,
        wq,
        scales,
        biases,
        rhs_indices=idx,
        transpose=True,
        group_size=group_size,
        bits=bits,
        sorted_indices=True,
    )
    assert _same(got, want)
    assert _proof(_TILE_PROOF).ran


@needs_nax
def test_the_rows_are_the_same_under_every_plan(toy, monkeypatch):
    toy(_tile_toy("plans"))
    tensors = _stack(128, 256, seed=4)
    x, idx = _rows(256)
    want = _want(tensors, x, idx)
    for plan in (
        nax.Plan(nax._SCHED_SEG, 64, 64, 0, 0),
        nax.Plan(nax._SCHED_DB, 96, 64, 32, 0),
        nax.Plan(nax._SCHED_SEG, 128, 128, 32, 8192),
    ):
        monkeypatch.setattr(nax, "face_plan", lambda *_, plan=plan: plan)
        layer = FormatSwitchLinear("tiletoy", 8, 1, 256, tensors)
        assert _same(layer(x, idx, sorted_indices=True), want)


def test_an_unsorted_call_of_the_same_shapes_runs_the_rows_kernel(toy):
    toy(_tile_toy("unsorted"))
    tensors = _stack(128, 256, seed=5)
    layer = FormatSwitchLinear("tiletoy", 8, 1, 256, tensors)
    other = FormatSwitchLinear("tiletoy", 8, 1, 256, tensors)
    x, idx = _rows(256)
    mx.eval(layer(x, idx, sorted_indices=True))
    rows = other(x, idx)
    assert _same(layer(x, idx), rows)
    assert _proof("format_switch_linear").ran


def test_a_sorted_call_after_an_unsorted_one_of_the_same_shapes_takes_the_tile_route(
    toy,
):
    toy(_tile_toy("after"))
    tensors = _stack(128, 256, seed=11)
    layer = FormatSwitchLinear("tiletoy", 8, 1, 256, tensors)
    x, idx = _rows(256)
    mx.eval(layer(x, idx))
    assert _same(layer(x, idx, sorted_indices=True), _want(tensors, x, idx))
    assert _proof(_TILE_PROOF).ran


def test_activations_the_tile_face_is_not_exact_for_leave_the_route():
    k, n = 256, 128
    w = mx.zeros((E, n, k), mx.bfloat16)
    wq, scales, biases = mx.quantize(w, group_size=64, bits=4)
    tensors = {"weight": wq, "scales": scales, "biases": biases}
    layer = FormatSwitchLinear("affine", 4, 64, k, tensors)
    x, idx = _rows(k, dtype=mx.float16)
    mx.eval(layer(x, idx, sorted_indices=True))
    assert not _proof(_TILE_PROOF).ran


@pytest.mark.parametrize(
    "counts",
    [
        (2, 1, 3, 2, 1, 1, 2, 1, 2, 0),  # 15 rows: below 16
        (3, 3, 3, 3, 3, 3, 3, 3, 3, 3),  # 3 an expert: below 4
    ],
)
def test_below_mlx_s_threshold_the_rows_kernel(toy, counts):
    toy(_tile_toy("few"))
    tensors = _stack(128, 256, seed=6)
    layer = FormatSwitchLinear("tiletoy", 8, 1, 256, tensors)
    x, idx = _rows(256, counts)
    got = layer(x, idx, sorted_indices=True)
    assert _same(got, FormatSwitchLinear("tiletoy", 8, 1, 256, tensors)(x, idx))
    assert not [k for k in kernels._PROOFS if k[0] == _TILE_PROOF]


@needs_nax
def test_an_n_the_tile_kernels_do_not_take_runs_the_rows_kernel(toy):
    toy(_tile_toy("ragged"))
    tensors = _stack(40, 256, seed=7)
    layer = FormatSwitchLinear("tiletoy", 8, 1, 256, tensors)
    x, idx = _rows(256)
    got = layer(x, idx, sorted_indices=True)
    assert _same(got, FormatSwitchLinear("tiletoy", 8, 1, 256, tensors)(x, idx))
    assert not [k for k in kernels._PROOFS if k[0] == _TILE_PROOF]


def test_a_tile_face_that_fails_leaves_the_rows_kernel(toy, caplog):
    toy(_tile_toy("broken", _TILE.replace("RUN = 8;", "RUN = 16;")))
    tensors = _stack(128, 256, seed=8)
    layer = FormatSwitchLinear("tiletoy", 8, 1, 256, tensors)
    x, idx = _rows(256)
    with caplog.at_level(logging.INFO, logger="omlx.formats.linear"):
        got = layer(x, idx, sorted_indices=True)
        again = layer(x, idx, sorted_indices=True)
    rows = FormatSwitchLinear("tiletoy", 8, 1, 256, tensors)(x, idx)
    assert _same(got, rows) and _same(again, rows)
    failed = "quant format face tiletoy_broken: tile kernel failed; rows kernel"
    assert caplog.text.count(failed) == 1
    assert _proof(_TILE_PROOF).failed
    assert _proof("format_switch_linear").ran


def test_the_tile_route_inside_mx_compile(toy):
    toy(_tile_toy("compiled"))
    tensors = _stack(128, 256, seed=9)
    x, idx = _rows(256)
    want = _want(tensors, x, idx)
    layer = FormatSwitchLinear("tiletoy", 8, 1, 256, tensors)
    compiled = mx.compile(lambda x, idx: layer(x, idx, sorted_indices=True))
    assert _same(compiled(x, idx), want)
    assert _proof(_TILE_PROOF).ran


def test_two_threads_through_one_layer_s_tile_route(toy):
    toy(_tile_toy("threads"))
    tensors = _stack(128, 256, seed=10)
    x, idx = _rows(256)
    want = _want(tensors, x, idx)
    mx.eval(want, *tensors.values())
    layer = FormatSwitchLinear("tiletoy", 8, 1, 256, tensors)

    def run(_):
        y = layer(x, idx, sorted_indices=True)
        mx.eval(y)
        return y

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, range(4)))
    assert all(_same(y, want) for y in results)


def test_expert_offload_keeps_a_format_stack_resident():
    # Offload reads MLX's quantized switch layers alone: a format stack is
    # skipped in words, and stays resident (where its prefill takes the
    # tile route).
    layer = FormatSwitchLinear("affine", 4, 64, 256, _affine_tensors())
    glu = SimpleNamespace(gate_proj=layer, up_proj=layer, down_proj=layer)
    view, reason = moe_expert_offload._resolve_store_view(glu, None, "layers.0.mlp")
    assert view is None and reason == "gate_proj is not a QuantizedSwitchLinear"


def _affine_tensors():
    w = mx.zeros((E, 128, 256), mx.bfloat16)
    wq, scales, biases = mx.quantize(w, group_size=64, bits=4)
    return {"weight": wq, "scales": scales, "biases": biases}


def test_two_layers_sharing_a_tile_face_choose_their_own_column_plan(toy):
    toy(_tile_toy("shared_width"))
    x, idx = _rows(256)
    for n in (128, 96):
        tensors = _stack(n, 256, seed=n)
        layer = FormatSwitchLinear("tiletoy", 8, 1, 256, tensors)
        assert _same(layer(x, idx, sorted_indices=True), _want(tensors, x, idx))


def test_a_warm_sorted_graph_cannot_hide_a_layers_wrong_input_width(toy):
    toy(_tile_toy("shared_k"))
    first = FormatSwitchLinear("tiletoy", 8, 1, 256, _stack(128, 256))
    wider = FormatSwitchLinear("tiletoy", 8, 1, 512, _stack(128, 512))
    x, idx = _rows(256)
    mx.eval(first(x, idx, sorted_indices=True))
    with pytest.raises(ValueError, match="512"):
        mx.eval(wider(x, idx, sorted_indices=True))


def test_tile_and_dot_faces_can_read_different_scale_operands(toy):
    source = _TILE.replace("return {r[0]};", "return {2.0f * r[0]};")
    face = replace(_tile_toy("own_scales", source), tensors=("tile_scales",))
    fmt = toy(face)
    fmt.tensors = ("group_scales", "tile_scales")
    x, idx = _rows(256)
    for seed in (53, 54):
        tensors = _stack(128, 256, seed=seed)
        tensors["tile_scales"] = tensors["group_scales"] * 0.5
        layer = FormatSwitchLinear("tiletoy", 8, 1, 256, tensors)
        assert _same(layer(x, idx, sorted_indices=True), _want(tensors, x, idx))
    assert _proof(_TILE_PROOF).ran


def test_stock_supports_declines_a_scalar_activation():
    tensors = _affine_tensors()
    scalar = mx.array(0, dtype=mx.bfloat16)
    assert not nax.supports(
        scalar,
        tensors["weight"],
        tensors["scales"],
        tensors["biases"],
        mx.zeros((16,), mx.uint32),
        64,
        4,
        "affine",
    )


def test_a_sorted_trace_uses_the_current_layers_operands(toy):
    toy(_tile_toy("trace_state"))
    tensors = _stack(128, 256, seed=57)
    layer = FormatSwitchLinear("tiletoy", 8, 1, 256, tensors)
    x, idx = _rows(256)
    call = mx.compile(
        lambda x, idx: layer(x, idx, sorted_indices=True), inputs=layer.state
    )
    assert _same(call(x, idx), _want(tensors, x, idx))
    tensors["group_scales"] = -tensors["group_scales"]
    layer["group_scales"] = tensors["group_scales"]
    want = _want(tensors, x, idx)
    assert _same(call(x, idx), want)
    assert _same(layer(x, idx, sorted_indices=True), want)


@pytest.mark.parametrize(
    "counts, takes_tile",
    [((5, 5, 5), False), ((4, 4, 4, 4), True), ((4, 4, 4, 4, 3), False)],
    ids=["fifteen", "sixteen", "below-four-per-expert"],
)
def test_sorted_rows_at_the_two_mlx_thresholds(toy, counts, takes_tile):
    toy(_tile_toy("thresholds"))
    tensors = {name: a[: len(counts)] for name, a in _stack(128, 256).items()}
    layer = FormatSwitchLinear("tiletoy", 8, 1, 256, tensors)
    x, idx = _rows(256, counts)
    got = layer(x, idx, sorted_indices=True)
    want = _want(tensors, x, idx) if takes_tile else layer(x, idx)
    assert _same(got, want)
    proved = [p for k, p in kernels._PROOFS.items() if k[0] == _TILE_PROOF]
    assert bool(proved and proved[0].ran) is takes_tile


@needs_nax
def test_the_number_of_experts_selects_the_sorted_graph(toy):
    toy(_tile_toy("expert_plans"))
    tensors = _stack(128, 256)
    for experts in (4, 10):
        stack = {name: a[:experts] for name, a in tensors.items()}
        layer = FormatSwitchLinear("tiletoy", 8, 1, 256, stack)
        # The same 480 rows require different schedules at 120 and 48 per expert.
        x, idx = _rows(256, (480 // experts,) * experts)
        assert _same(layer(x, idx, sorted_indices=True), _want(stack, x, idx))
    assert len(_proof(_TILE_PROOF).ran) == 2


@needs_steel
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_ragged_columns_use_the_toy_face_on_steel(toy, dtype):
    toy(_tile_toy("steel_columns"))
    k, n = 256, 40
    tensors = _stack(n, k, seed=79)
    layer = FormatSwitchLinear("tiletoy", 8, 1, k, tensors)
    x, idx = _rows(k, dtype=dtype)
    got = layer(x, idx, sorted_indices=True)
    assert _same(got, _want(tensors, x, idx))
    assert _proof(_TILE_PROOF).ran


@needs_steel
def test_a_48_wide_call_keeps_the_rows_kernel_on_steel(toy):
    toy(_tile_toy("steel_short_k"))
    k, n = 48, 40
    tensors = _stack(n, k, seed=83)
    layer = FormatSwitchLinear("tiletoy", 8, 1, k, tensors)
    x, idx = _rows(k)
    got = layer(x, idx, sorted_indices=True)
    assert _same(got, layer(x, idx))
    assert not any(key[0] == _TILE_PROOF for key in kernels._PROOFS)


@needs_steel
def test_steel_affine_partial_output_tile_matches_mlx():
    k, n = 256, 40
    mx.random.seed(73)
    w = mx.random.normal((E, n, k)).astype(mx.bfloat16)
    codes, scales, biases = mx.quantize(w, group_size=32, bits=8)
    tensors = {"weight": codes, "scales": scales, "biases": biases}
    layer = FormatSwitchLinear("affine", 8, 32, k, tensors)
    x, ids = _rows(k)
    got = layer(x, ids, sorted_indices=True)
    want = mx.gather_qmm(
        x,
        codes,
        scales,
        biases,
        rhs_indices=ids,
        transpose=True,
        group_size=32,
        bits=8,
        sorted_indices=True,
    )
    assert _same(got, want)
    assert _proof(_TILE_PROOF).ran


@needs_steel
def test_splitting_a_sorted_run_at_an_unaligned_row_keeps_its_bits(toy):
    toy(_tile_toy("split_run"))
    tensors = _stack(40, 256)
    layer = FormatSwitchLinear("tiletoy", 8, 1, 256, tensors)
    x, ids = _rows(256)
    left = layer(x[:53], ids[:53], sorted_indices=True)
    right = layer(x[53:], ids[53:], sorted_indices=True)
    want = _want(tensors, x, ids)
    assert _same(mx.concatenate((left, right)), want)
    assert _same(layer(x, ids, sorted_indices=True), want)
    assert _proof(_TILE_PROOF).ran


@needs_steel
def test_a_sparse_expert_uses_the_full_weight_row_offset():
    from omlx.formats import tile_gather

    # Only the first 32 outputs for expert four are dispatched. This face
    # derives a weight from its logical row and reads no backing weight page.
    source = """
    struct WideRow {
      static constant constexpr int CONTRACT = 1;
      const device uchar* codes;
      template <int K> struct Runs {
        static constant constexpr int RUN = 8;
        static constant constexpr int GROUP = 16;
        using P = float;
        using Codes = uint;
        size_t row;
        Runs(WideRow, size_t r) : row(r) {}
        P params(int) const {
          return float((row >> 32) == 1) + 8.0f * float((row & 31) == 0);
        }
        Codes fetch(int) const { return 0; }
        __OMLX_READ_WEIGHT__ { return p; }
      };
    };
    """
    face = Tile(
        key="steel_wide_row",
        type="WideRow",
        source=source,
        tensors=(),
        reads=formats.TileReads(),
    )
    kernel = tile_gather.kernel(face)
    x = mx.ones((1, 1, 32), mx.float16)
    dummy = mx.zeros((8,), mx.uint8)
    offsets = mx.zeros((8,), mx.int32)
    for n in (40, (1 << 30) + 8):
        # Five experts, the first four with zero rows; M=1 and K=32.
        meta = mx.array([1, n, 5], dtype=mx.int32)
        (out,) = kernel(
            inputs=[x, dummy, offsets, meta],
            template=[("T", x.dtype), ("K", 32), ("ALIGN_N", False)],
            grid=(64, 1, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(1, 1, 32)],
            output_dtypes=[x.dtype],
        )
        rows = range(4 * n, 4 * n + 32)
        expected = [32 * ((r >> 32 == 1) + 8 * (r % 32 == 0)) for r in rows]
        assert out.tolist() == [[expected]]


def test_a_sorted_switch_builds_its_tile_once(toy, monkeypatch):
    from omlx.formats import linear

    k, n = 256, 64
    toy(_tile_toy("sorted_once"))
    tensors = _stack(n, k)
    layer = FormatSwitchLinear("tiletoy", 8, 1, k, tensors)
    x, indices = _rows(k)
    built = []
    build = linear._tile

    def record(held):
        built.append(held is layer)
        return build(held)

    monkeypatch.setattr(linear, "_tile", record)
    for _ in range(2):
        assert _same(layer(x, indices, sorted_indices=True), _want(tensors, x, indices))
    assert built == [True]


@needs_steel
def test_a_sorted_format_layer_stages_its_faces_table(monkeypatch):
    from omlx.formats import affine
    from tests.test_formats_tables import tabled_tile

    face = tabled_tile(affine.tile_face(4, 64, mx.bfloat16))
    monkeypatch.setattr(affine.AffineFormat, "tile", lambda *args: face)
    # Make decoding require distinct lookup-table entries.
    mx.random.seed(91)
    w = mx.random.normal((E, 128, 256)).astype(mx.bfloat16)
    codes, scales, biases = mx.quantize(w, 64, 4)
    tensors = dict(weight=codes, scales=scales, biases=biases)
    layer = FormatSwitchLinear("affine", 4, 64, 256, tensors)
    x, idx = _rows(256)
    got = layer(x, idx, sorted_indices=True)
    want = mx.gather_qmm(
        x,
        codes,
        scales,
        biases,
        rhs_indices=idx,
        transpose=True,
        group_size=64,
        bits=4,
        sorted_indices=True,
    )
    assert _same(got, want)
    assert _proof(_TILE_PROOF).ran


@needs_steel
def test_missing_steel_headers_keep_two_layers_on_the_rows_kernel(
    toy, monkeypatch, caplog
):
    from functools import cache

    from omlx.formats import tile_gather

    reads = []

    def missing(paths):
        reads.append(paths)
        return None

    monkeypatch.setattr(tile_gather, "_read_mlx_headers", missing)
    monkeypatch.setattr(
        tile_gather, "_headers", cache(tile_gather._headers.__wrapped__)
    )
    monkeypatch.setattr(tile_gather, "_build", cache(tile_gather._build.__wrapped__))
    x, ids = _rows(256)
    for key in ("first_missing", "second_missing"):
        toy(_tile_toy(key))
        layer = FormatSwitchLinear("tiletoy", 8, 1, 256, _stack(40, 256))
        assert _same(layer(x, ids, sorted_indices=True), layer(x, ids))
    assert len(reads) == 1
    assert caplog.text.count("MLX steel headers unavailable") == 1
    assert _proof("format_switch_linear").ran
    assert not any(key[0] == _TILE_PROOF for key in kernels._PROOFS)


@needs_steel
@pytest.mark.parametrize("change", ["signed_ids", "fp32", "rank_two_ids"])
def test_steel_declines_calls_outside_the_sorted_layout(toy, change):
    toy(_tile_toy("steel_layout"))
    layer = FormatSwitchLinear("tiletoy", 8, 1, 256, _stack(40, 256))
    x, ids = _rows(256)
    if change == "signed_ids":
        ids = ids.astype(mx.int32)
    elif change == "fp32":
        x = x.astype(mx.float32)
    else:
        ids = ids.reshape(1, -1)
    assert _same(layer(x, ids, sorted_indices=True), layer(x, ids))
    assert not any(key[0] == _TILE_PROOF for key in kernels._PROOFS)


@needs_steel
def test_steel_reads_both_groups_of_a_threads_sixteen_weights():
    from omlx.formats import tile_gather

    # Flip the sign on alternating eight-weight groups. The two groups
    # owned by a loader thread need different parameters from the same row.
    face = _tile_toy(
        "two_groups",
        _TILE.replace(
            "P params(int) const {\n      return {r[0]};",
            "P params(int g) const {\n      return {(g & 1) ? -r[0] : r[0]};",
        ),
    )
    k = 256
    tensors = _stack(40, k, seed=97)
    x, ids = _rows(k)
    got = tile_gather.launch(
        tile_gather.kernel(face), x, (tensors["codes"], tensors["group_scales"]), ids
    )
    signs = mx.where((mx.arange(k) // 8) % 2 == 0, 1.0, -1.0)
    values = tensors["codes"].astype(mx.float32) - 128.0
    w = (values * tensors["group_scales"][..., None] * signs).astype(x.dtype)
    want = mx.gather_mm(x, w.swapaxes(-1, -2), rhs_indices=ids, sorted_indices=True)
    assert _same(got, want)
