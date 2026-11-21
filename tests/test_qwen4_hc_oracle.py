# SPDX-License-Identifier: Apache-2.0
"""The live hyper-connection kernels against their frozen oracle (tests/oracles).

Every weight-reading kernel of ``hc_fused.py`` is compared with the oracle bit
for bit: the two-launch decode at 1 to 6 rows, the three-launch decode at 7 to
16, the prefill's tail kernel above 16, with and without the inject rows and a
pending write, at 4, 5, 6 and 8 bits, groups 32 and 64, BF16 and FP16, in both
GPU classes' geometry, at a hidden size whose K slices end in a partial block
and at sizes whose do not, and on all-zero rows. FP32 activations run on no
served path, but there the outputs keep every FP32 rounding of the sums, which
the 16-bit outputs mostly absorb: they show a change of summation order
(hc_fused takes FP32 rows in its decode, without a pending write).
"""

from __future__ import annotations

import ast
import inspect
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import replace
from threading import Barrier

import mlx.core as mx
import pytest

from omlx.formats import affine
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from tests.oracles import hc_fused as oracle
from tests.test_qwen4_hc_fused import HC, LOWRANK, _module

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture(autouse=True)
def _vendored_qwen4():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()


def _live():
    from mlx_vlm.models.qwen4_exp import hc_fused

    return hc_fused


# Each source the oracle runs, by the live module and name it is kept under.
_SOURCES = {
    "HEADER": ("fused", "_HEADER"),
    "TG_HEADER": ("fused", "_TG_HEADER"),
    "N_SOURCE": ("fused", "_N_SOURCE"),
    "NP_SOURCE": ("fused", "_NP_SOURCE"),
    "WN_SOURCE": ("fused", "_WN_SOURCE"),
    "WND_SOURCE": ("fused", "_WND_SOURCE"),
    "TI_SOURCE": ("fused", "_TI_SOURCE"),
}


@pytest.mark.parametrize("name", sorted(_SOURCES))
def test_oracle_sources_are_the_live_ones(name):
    module, live = _SOURCES[name]
    source = getattr(_live(), live)
    assert getattr(oracle, name) == source


def test_oracle_sources_are_literals():
    """The oracle keeps each source as text, not as a reference to the live
    module, so a port that changes a live source cannot change the oracle."""
    tree = ast.parse(inspect.getsource(oracle))
    literals = {
        node.targets[0].id
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    derived = {"NP_SOURCE", "WND_SOURCE", "NDN_SOURCE", "NDNW_SOURCE", "V2_HEADER"}
    pieces = {"NDN_SOURCE_TEMPLATE", "NDN_LOAD", "NDN_WRITE_LOAD"}
    frozen_decode = {"D_SOURCE", "U_SOURCE", "U2_SOURCE"}
    assert (set(_SOURCES) - derived) | pieces | frozen_decode <= literals
    imported = {
        name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for name in (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
        )
    }
    assert imported == {"__future__", "mlx"}


def _view(dtype):
    return {mx.float32: mx.uint32, mx.bfloat16: mx.uint16, mx.float16: mx.uint16}[dtype]


def _assert_bitwise(live, expected, what):
    live = live if isinstance(live, tuple) else (live,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    assert len(live) == len(expected), what
    mx.eval(live, expected)
    for i, (a, b) in enumerate(zip(live, expected)):
        assert a.shape == b.shape and a.dtype == b.dtype, (what, i)
        view = _view(a.dtype)
        assert mx.array_equal(a.view(view), b.view(view)).item(), (what, i)


def _inputs(rows, hidden, dtype, seed, zero=None, batch=1):
    mx.random.seed(seed)
    shape = (batch, rows, HC * hidden)
    if zero is None:
        hyper = (mx.random.normal(shape) * 2).astype(dtype)
    else:
        hyper = mx.full(shape, zero, dtype=dtype)
    branch = mx.random.normal((batch, rows, hidden)).astype(dtype)
    gate = (2 * mx.sigmoid(mx.random.normal((batch, rows, HC)))).astype(dtype)
    if zero is not None:
        branch, gate = mx.full(branch.shape, zero, dtype=dtype), mx.zeros_like(gate)
    mx.eval(hyper, branch, gate)
    return hyper, (branch, gate)


def _modules(bits, group_size, dtype, hidden=2560, seed=0, lowrank=LOWRANK):
    """A module with the inject rows and one without, the same down and up."""
    mx.random.seed(seed)
    with_inject = _module(
        bits, True, hidden=hidden, group_size=group_size, dtype=dtype, lowrank=lowrank
    )
    without = copy(with_inject)
    del without["block_inject_weight"]
    return with_inject, without


def _decode(monkeypatch, module, hyper, write, gpu_class):
    hc_fused = _live()
    monkeypatch.setattr(hc_fused, "_gpu_class", lambda: gpu_class)
    if hasattr(getattr(hc_fused, "_decode_launch", None), "__wrapped__"):
        monkeypatch.setattr(
            hc_fused, "_decode_launch", hc_fused._decode_launch.__wrapped__
        )
    live = hc_fused.fused_forward(module, hyper, write)
    assert live is not None
    return live, oracle.decode(module, hyper, write, gpu_class=gpu_class)


def test_decode_helper_switches_gpu_class_geometry(monkeypatch):
    """The helper's class switch rebuilds the launches: the 's' geometry runs."""
    hidden, dtype = 256, mx.bfloat16
    mx.random.seed(2564)
    module = _module(8, False, hidden=hidden)
    hyper = (mx.random.normal((1, 4, HC * hidden)) * 2).astype(dtype)
    live = _live()
    seen = []
    original = live._down_rps

    def record(rows):
        seen.append(original(rows))
        return seen[-1]

    monkeypatch.setattr(live, "_down_rps", record)
    _decode(monkeypatch, module, hyper, None, "d")
    _decode(monkeypatch, module, hyper, None, "s")
    assert set(seen) == {2, 4}, seen


def _assert_decode(
    monkeypatch, modules, hidden, dtype, rows_list, gpu_class, seed, writes=True
):
    for module in modules:
        for rows in rows_list:
            if rows > oracle.V2_MAX_ROWS and gpu_class != "d":
                continue  # the three-launch kernels have one geometry
            hyper, pending = _inputs(rows, hidden, dtype, seed + rows)
            for write in (None, pending) if writes else (None,):
                what = ("decode", rows, "block_inject_weight" in module, write is None)
                _assert_bitwise(
                    *_decode(monkeypatch, module, hyper, write, gpu_class), what
                )


@pytest.mark.parametrize("gpu_class", ["d", "s"])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_decode_matches_the_oracle(bits, group_size, dtype, gpu_class, monkeypatch):
    """1 to 6 rows take the two-launch kernels, 7 to 16 the three-launch ones.
    The rows are those of each geometry: down rows per simdgroup and up rows
    per threadgroup change at 2, 3 and 4 rows, and not above 7."""
    seed = 1500 + 10 * bits + group_size
    modules = _modules(bits, group_size, dtype, seed=seed)
    rows = (1, 2, 3, 4, 5, 6, 7, 16)
    _assert_decode(monkeypatch, modules, 2560, dtype, rows, gpu_class, seed)


# 1152: every K slice's last down block is partial (128 of 512 or 256 values),
# and the inject's at 4 and 5 bits; 1344: 320 or 64 down, 64 inject; the norm's
# last 256-element strip is partial in both. Groups of 32 put a partial block's
# lanes in more than one group; FP32 rows show its summation order.
@pytest.mark.parametrize("gpu_class", ["d", "s"])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("hidden", [1152, 1344])
def test_decode_with_partial_blocks_matches_the_oracle(
    hidden, bits, group_size, dtype, gpu_class, monkeypatch
):
    seed = 1600 + hidden + bits + group_size
    modules = _modules(bits, group_size, dtype, hidden=hidden, seed=seed)
    rows = (1, 2, 3, 4, 6, 7, 16)
    # hc_fused takes no pending write of FP32 rows.
    writes = dtype != mx.float32
    _assert_decode(monkeypatch, modules, hidden, dtype, rows, gpu_class, seed, writes)


@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_float32_decode_matches_the_oracle(bits, group_size, monkeypatch):
    seed = 1700 + 10 * bits + group_size
    modules = _modules(bits, group_size, mx.float32, seed=seed)
    rows = range(1, oracle.MAX_ROWS + 1)
    for gpu_class in ("d", "s"):
        # hc_fused takes no pending write of FP32 rows.
        _assert_decode(
            monkeypatch, modules, 2560, mx.float32, rows, gpu_class, seed, False
        )


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("zero", [0.0, -0.0])
def test_zero_rows_match_the_oracle(zero, bits, dtype, monkeypatch):
    """All-zero rows (and pending writes of zeros): each sum is of zeros, whose
    sign is the arithmetic's."""
    modules = _modules(bits, 64, dtype, seed=1800 + bits)
    for module in modules:
        for rows in (1, 7):
            hyper, pending = _inputs(rows, 2560, dtype, 0, zero=zero)
            for write in (None, pending) if dtype != mx.float32 else (None,):
                for gpu_class in ("d", "s") if rows <= oracle.V2_MAX_ROWS else ("d",):
                    _assert_bitwise(
                        *_decode(monkeypatch, module, hyper, write, gpu_class),
                        ("zero decode", rows),
                    )
        if dtype == mx.float32:
            continue  # hc_fused's prefill takes 16-bit rows only
        hyper, pending = _inputs(17, 2560, dtype, 0, zero=zero)
        for write in (None, pending):
            live = _live().prefill_forward(module, hyper, write, nax=False)
            assert live is not None
            _assert_bitwise(live, oracle.prefill(module, hyper, write), "zero prefill")


# 2560: the tail kernel takes the inject rows; 1152: its width is not whole
# packs of 256 threads, and the compiled tail and MLX's inject take them.
@pytest.mark.parametrize("hidden", [2560, 1152])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_prefill_matches_the_oracle(bits, group_size, dtype, hidden):
    seed = 1900 + 10 * bits + group_size + hidden
    for module in _modules(bits, group_size, dtype, hidden=hidden, seed=seed):
        for rows in (17, 300):
            hyper, pending = _inputs(rows, hidden, dtype, seed + rows)
            for write in (None, pending):
                live = _live().prefill_forward(module, hyper, write, nax=False)
                assert live is not None
                _assert_bitwise(
                    live,
                    oracle.prefill(module, hyper, write),
                    ("prefill", rows, "block_inject_weight" in module, write is None),
                )


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_the_two_decode_forms_give_a_row_the_same_bits(bits, group_size, dtype):
    """What the ports keep: wherever the two-launch kernels take a call, at
    any row count, they give each output the three-launch kernels' bits."""
    seed = 2000 + 10 * bits + group_size
    modules = _modules(bits, group_size, dtype, seed=seed)
    for rows in range(1, oracle.MAX_ROWS + 1):
        hyper, pending = _inputs(rows, 2560, dtype, seed + rows)
        for module in modules:
            inject = "block_inject_weight" in module
            for write in (None, pending):
                expected = oracle.decode(
                    module, hyper, write, gpu_class="d", two_launch=False
                )
                for gpu_class in ("d", "s"):
                    assert oracle.two_launch_fits(
                        rows, HC, 2560, LOWRANK, bits, inject, gpu_class, dtype.size
                    )
                    _assert_bitwise(
                        oracle.decode(
                            module, hyper, write, gpu_class=gpu_class, two_launch=True
                        ),
                        expected,
                        ("forms", gpu_class, rows, inject, write is None),
                    )


# Shapes on each side of every clause of the two-launch fit: three streams;
# a low rank of whole row groups or not (328, 336); the norm's strips against the
# row groups (5376, 5440); the stream in threadgroup memory (16320, 16384, and
# 16368 exactly at the limit); the up tile's threadgroup memory (512 at 8 bits
# over it; 960, over it only with its activations and the staging's padding
# column; 3264 at 4 bits and 32 outputs, exactly at the limit; 2560).
_FORM_SHAPES = [
    (hc, hidden, lowrank)
    for hc in (3, 4)
    for hidden in (2560, 5376, 5440, 16320, 16368, 16384)
    for lowrank in (320, 328, 336, 384, 512, 960, 2560, 3264)
]


def test_the_oracle_takes_hc_fused_s_form(monkeypatch):
    """The oracle's choice of form is hc_fused's: two launches to 6 rows where
    they fit, three above, and three where the two do not fit."""
    hc_fused = _live()
    seen = set()
    for gpu_class in ("d", "s"):
        monkeypatch.setattr(hc_fused, "_gpu_class", lambda c=gpu_class: c)
        for hc, hidden, lowrank in _FORM_SHAPES:
            for rows in range(1, oracle.MAX_ROWS + 1):
                for bits in (4, 5, 6, 8):
                    for inject in (True, False):
                        chunks = oracle.up_chunks(bits, lowrank)
                        live = hc_fused._decode_v2(
                            rows, hc, hidden, lowrank, chunks, inject
                        )
                        fits = oracle.two_launch_fits(
                            rows, hc, hidden, lowrank, bits, inject, gpu_class
                        )
                        seen.add((rows <= oracle.V2_MAX_ROWS, fits))
                        assert live == oracle.takes_two_launches(
                            rows, hc, hidden, lowrank, bits, inject, gpu_class
                        ), (gpu_class, hc, hidden, lowrank, rows, bits, inject)
    assert seen == {(True, True), (True, False), (False, True), (False, False)}


def test_decode_affine_partial_block_requires_a_whole_group():
    """96 values fill whole 8-bit lanes but truncate the second group of 64."""
    from omlx.formats import testing

    face = affine.face(8, 64, True, mx.bfloat16, partial_blocks=True)
    x = mx.zeros((96,), mx.bfloat16)
    codes = mx.zeros((8, 24), mx.uint32)
    params = mx.ones((8, 2), mx.bfloat16)
    with pytest.raises(RuntimeError, match="K must be whole lanes and groups"):
        mx.eval(testing.dot_rows(face, [codes, params, params], x, 8))


def test_decode_refuses_an_empty_up_tile(monkeypatch):
    # Eight 4-bit weights do not fill a two-pack lane. No tile can launch.
    chunks = oracle.up_chunks(4, 8)
    assert chunks == 0
    with pytest.raises(ValueError, match="no whole up chunks"):
        oracle.up_threads(4, 8)
    assert not oracle.two_launch_fits(1, HC, 256, 8, 4, False, "d")
    live = _live()
    monkeypatch.setattr(live, "_gpu_class", lambda: "d")
    assert not live._decode_v2(1, HC, 256, 8, chunks, False)


def test_the_oracle_s_geometry_is_hc_fused_s(monkeypatch):
    """Geometry changes no output bit, so no output comparison can see it: the
    oracle's constants and launch shapes are compared with hc_fused's."""
    hc_fused = _live()
    assert (
        oracle.MAX_ROWS,
        oracle.V2_MAX_ROWS,
        oracle.V2_UP_THREADS,
        oracle.TG_MEMORY_BYTES,
    ) == (
        hc_fused.MAX_ROWS,
        hc_fused._V2_MAX_ROWS,
        hc_fused._V2_UP_THREADS,
        hc_fused._TG_MEMORY_BYTES,
    )
    for gpu_class in ("d", "s"):
        monkeypatch.setattr(hc_fused, "_gpu_class", lambda c=gpu_class: c)
        for rows in range(1, oracle.MAX_ROWS + 1):
            assert oracle.down_rps(rows, gpu_class) == hc_fused._down_rps(rows)
            assert oracle.up_outputs(rows, gpu_class) == hc_fused._up_outputs(rows)
    for bits in (4, 5, 6, 8):
        assert oracle.pack_factor(bits) == hc_fused._pack_factor(bits)
        for lowrank in (320, 384, 512):
            face = affine.face(bits, 64, True, mx.bfloat16, partial_blocks=True)
            chunks = lowrank // face.vpt
            assert oracle.up_chunks(bits, lowrank) == chunks
            # hc_fused's two-launch up: whole chunk rows within its threads.
            threads = chunks * (hc_fused._V2_UP_THREADS // chunks)
            assert oracle.up_threads(bits, lowrank) == threads
    for lowrank, inject, rps in ((320, True, 1), (320, False, 2), (512, True, 4)):
        assert oracle.down_row_groups(
            lowrank, inject, rps
        ) == hc_fused._down_row_groups(lowrank, inject, rps)


def _strided(array):
    """``array`` as a view of every other element of a buffer twice its size:
    a kernel input that is not row-contiguous."""
    doubled = mx.stack([array, mx.zeros_like(array)], axis=-1)
    view = doubled.reshape(*array.shape[:-1], 2 * array.shape[-1])[..., ::2]
    mx.eval(view)
    return view


@pytest.mark.parametrize("bits", [4, 8])
def test_batched_and_strided_rows_match_the_oracle(bits, monkeypatch):
    """Rows of several sequences (batch above 1), and a norm weight and rows
    that are views of every other element: what a kernel reads must be the
    logical elements. The oracle's kernels are its own, by name."""
    module, without = _modules(bits, 64, mx.bfloat16, seed=2200 + bits)
    module.hc_norm.weight = _strided(module.hc_norm.weight)
    for batch, seq in ((2, 3), (3, 5), (2, 9)):
        hyper, pending = _inputs(seq, 2560, mx.bfloat16, 2200 + seq, batch=batch)
        strided = _strided(hyper)
        strided_pending = tuple(_strided(a) for a in pending)
        for m in (module, without):
            for rows_in in (hyper, strided):
                for write in (None, pending, strided_pending):
                    what = ("batched", batch, seq, write is None)
                    if batch * seq <= oracle.MAX_ROWS:
                        classes = (
                            ("d", "s") if batch * seq <= oracle.V2_MAX_ROWS else ("d",)
                        )
                        for gpu_class in classes:
                            _assert_bitwise(
                                *_decode(monkeypatch, m, rows_in, write, gpu_class),
                                what,
                            )
                    else:
                        live = _live().prefill_forward(m, rows_in, write, nax=False)
                        assert live is not None
                        _assert_bitwise(live, oracle.prefill(m, rows_in, write), what)
    assert oracle._KERNELS
    assert all(name.startswith("omlx_oracle_") for name in oracle._KERNELS)


# Each projection's width and group size are its own template values.
@pytest.mark.parametrize(
    "down,up,inject",
    [
        ((4, 64), (8, 32), (8, 64)),
        ((8, 32), (5, 64), (6, 32)),
        ((6, 64), (4, 64), (5, 32)),
    ],
)
def test_mixed_projections_match_the_oracle(down, up, inject, monkeypatch):
    mx.random.seed(2100 + down[0] + 10 * up[0] + 100 * inject[0])
    module = _module(down[0], True, group_size=down[1])
    module.input_mix_weight_up = _module(
        up[0], False, group_size=up[1]
    ).input_mix_weight_up
    module.block_inject_weight = _module(
        inject[0], True, group_size=inject[1]
    ).block_inject_weight
    for gpu_class in ("d", "s"):
        rows = range(1, oracle.MAX_ROWS + 1)
        _assert_decode(monkeypatch, [module], 2560, mx.bfloat16, rows, gpu_class, 2100)
    for rows in (17, 300):
        hyper, pending = _inputs(rows, 2560, mx.bfloat16, 2100 + rows)
        for write in (None, pending):
            live = _live().prefill_forward(module, hyper, write, nax=False)
            assert live is not None
            _assert_bitwise(
                live, oracle.prefill(module, hyper, write), ("prefill", rows)
            )


@pytest.mark.parametrize(
    "projection",
    ["input_mix_weight_down", "block_inject_weight", "input_mix_weight_up"],
)
@pytest.mark.parametrize("rows,write", [(1, False), (3, True), (7, False)])
def test_decode_uses_each_projections_dot_face(projection, rows, write, monkeypatch):
    """Encode one nonzero per row twice. The toy reads centered bytes and
    finishes with a row scale. The oracle reads unsigned affine bytes with
    that scale in every group. A single product makes both traversals exact.
    Each other projection retains its stock weights and traversal."""
    from tests.test_formats_kernels import _toy

    live = _live()
    stock_face = live._decode_face
    toy = replace(_toy("hc"), tensors=("hc_gains",))
    mx.random.seed(2500 + rows)
    hidden = 1152
    module = _module(8, True, hidden=hidden)
    layer = module[projection]
    n, words = layer.weight.shape
    k = words * 4
    # Row zero hits the last chunk. Following rows span different slices,
    # groups and lanes, including the down projection's partial last block.
    selected = (k - 1 - mx.arange(n) * 67) % k
    codes = (mx.arange(k)[None, :] == selected[:, None]).astype(mx.uint8)
    # MLX binds fewer than eight scalars as constants. Inject has four rows.
    gains = 2.0 ** ((mx.arange(max(n, 8)) % 4).astype(mx.float32) - 2)
    layer.weight = codes.view(mx.uint32)
    layer.scales = mx.broadcast_to(gains[:n, None], layer.scales.shape).astype(
        mx.bfloat16
    )
    layer.biases = mx.zeros_like(layer.scales)
    hyper, pending = _inputs(rows, hidden, mx.bfloat16, 2550 + rows)
    operands = pending if write else None
    expected = oracle.decode(module, hyper, operands, gpu_class="d")
    mx.eval(expected)

    layer.weight = (codes + 128).astype(mx.uint8).view(mx.uint32)
    layer.hc_gains = gains
    monkeypatch.setattr(live, "_gpu_class", lambda: "d")
    monkeypatch.setattr(
        live,
        "_decode_face",
        lambda p, inject: toy if p is layer else stock_face(p, inject),
    )
    result = live.fused_forward(module, hyper, operands)
    assert result is not None
    _assert_bitwise(result, expected, (projection, rows, write))


def test_decode_up_stages_each_chunk_before_adding_it(monkeypatch):
    """A fused multiply-add rounds differently with a running accumulator.
    The two-launch kernel stores each chunk in threadgroup memory. A serial
    up row must give that staged sum's bits even for a face using fma."""
    from omlx.formats import ApplyRead, Dot, DotReads, LoadRead, MaddRead, TypeParam

    live = _live()
    stock_face = live._decode_face
    toy = Dot(
        key="hc_fma",
        type="HcFma",
        tensors=(),
        vpt=4,
        source=r"""
struct HcFma {
    static constant constexpr int CONTRACT = 1;
    const device uint8_t* codes;
    template <int K> struct Lanes {
        static constant constexpr int VPT = 4, BLOCK = 32 * VPT;
        const device uint8_t* w;
        Lanes(HcFma m, size_t row, uint lane)
            : w(m.codes + row * K + lane * VPT) {}
        using Row = const device uint8_t*;
        Row row(int r) const { return w + r * K; }
        void step() { w += BLOCK; }
        __OMLX_READ_LOAD__ {
            for (int i = 0; i < VPT; ++i) xt[i] = float(x[i]);
            return 0.0f;
        }
        __OMLX_READ_MADD__ {
            float dot = 0.0f;
            for (int i = 0; i < VPT; ++i) dot += (float(r[i]) - 128.0f) * xt[i];
            return metal::fma(0.013f, dot, acc);
        }
        using Decoded = Row;
        __OMLX_READ_APPLY__ {
            return madd(acc, d, xt, aux);
        }
        float finish(int, float sum) const { return sum; }
    };
};
""",
        reads=DotReads(
            load=LoadRead(input=TypeParam("XP")),
            madd=MaddRead(params=("acc", "r", "xt", "")),
            apply=ApplyRead(),
        ),
    )
    mx.random.seed(2560)
    module = _module(8, True, hidden=256, dtype=mx.float32)
    up = module.input_mix_weight_up
    n = up.weight.shape[0]
    codes = mx.random.randint(126, 131, (n, LOWRANK)).astype(mx.uint8)
    up.weight = codes.view(mx.uint32)
    monkeypatch.setattr(
        live,
        "_decode_face",
        lambda p, inject: toy if p is up else stock_face(p, inject),
    )
    hyper, _ = _inputs(3, 256, mx.float32, 2561)
    monkeypatch.setattr(live, "_DECODE_V2", True)
    monkeypatch.setattr(live, "_gpu_class", lambda: "d")
    if hasattr(getattr(live, "_decode_launch", None), "__wrapped__"):
        monkeypatch.setattr(live, "_decode_launch", live._decode_launch.__wrapped__)
    staged = live.fused_forward(module, hyper)
    assert staged is not None
    mx.eval(staged)
    monkeypatch.setattr(live, "_DECODE_V2", False)
    serial = live.fused_forward(module, hyper)
    assert serial is not None
    _assert_bitwise(serial, staged, "up chunk accumulation")


@pytest.mark.parametrize("rows", [7, 16])
def test_affine_serial_up_streams_chunks_in_order(rows, monkeypatch):
    """Across two groups, (2**25 - 2**25) + (+/-1) must retain the last code."""
    hidden, dtype = 256, mx.bfloat16
    width = HC * hidden
    module = _module(8, False, hidden=hidden)
    module.hc_norm.weight = mx.zeros((width,), dtype)
    up = module.input_mix_weight_up
    codes = mx.zeros((width, LOWRANK), mx.uint8)
    for k in (0, 64, 73):
        codes[:, k] = 1
    up.weight = codes.view(mx.uint32)
    up.scales = mx.ones_like(up.scales)
    up.scales[:, 0] = 2
    up.biases = mx.zeros_like(up.biases)

    positive = mx.arange(rows) % 2 == 0
    act = mx.zeros((rows, LOWRANK), dtype)
    act[:, 0] = 2**24
    act[:, 64] = -(2**25)
    act[:, 73] = mx.where(positive, 1, -1).astype(dtype)

    live = _live()
    build = mx.fast.metal_kernel
    up_sources = []

    def with_known_activations(**spec):
        if "hc_fused_down" in spec["name"]:
            return lambda **kwargs: (act, mx.zeros((rows, HC), dtype))
        if "hc_fused_up" in spec["name"]:
            up_sources.append(spec["source"])
        return build(**spec)

    monkeypatch.setattr(mx.fast, "metal_kernel", with_known_activations)
    monkeypatch.setattr(live, "_KERNELS", {})
    monkeypatch.setattr(live, "_VALIDATED", set())
    if hasattr(live, "_decode_launch"):
        monkeypatch.setattr(live, "_decode_launch", live._decode_launch.__wrapped__)
        monkeypatch.setattr(live.kernels, "_FACE_KERNELS", {})

    mixed = live.fused_forward(module, mx.ones((1, rows, width), dtype))
    assert mixed is not None
    expected = mx.where(positive, 0.73046875, 0.26953125).astype(dtype)
    expected = mx.broadcast_to(expected.reshape(1, rows, 1), (1, rows, hidden))
    _assert_bitwise(mixed, expected, "affine group and chunk carry")
    assert len(up_sources) == 1
    assert "hc_stream_up<R>(up, n, a)" in up_sources[0], "up did not stream the group"


def _kernel_recorder(live, monkeypatch, rows, width, hidden, lowrank, dtype):
    """Record each kernel build; return its (name, header, source).

    Every build gets a canned output of the right shape and dtype, so a route
    can be read without compiling or running Metal.
    """
    builds = []

    def record(**spec):
        name = spec["name"]
        builds.append((name, spec.get("header", ""), spec.get("source", "")))
        if "fused_write_norm" in name:
            return lambda **kwargs: [
                mx.zeros((rows, width), dtype),
                mx.zeros((rows, width), dtype),
            ]
        if "fused_norm" in name:
            return lambda **kwargs: [mx.zeros((rows, width), dtype)]
        if "hc_fused_down" in name:
            return lambda **kwargs: [
                mx.zeros((rows, lowrank), dtype),
                mx.zeros((rows, HC), dtype),
            ]
        if "decode_write_norm_down" in name:
            return lambda **kwargs: [
                mx.zeros((rows, width), dtype),
                mx.zeros((rows, width), dtype),
                mx.zeros((rows, HC, lowrank + HC), mx.float32),
            ]
        if "decode_norm_down" in name:
            return lambda **kwargs: [
                mx.zeros((rows, width), dtype),
                mx.zeros((rows, HC, lowrank + HC), mx.float32),
            ]
        return lambda **kwargs: [
            mx.zeros((rows, hidden), dtype),
            mx.zeros((rows, HC), dtype),
        ]

    monkeypatch.setattr(mx.fast, "metal_kernel", record)
    monkeypatch.setattr(live, "_KERNELS", {})
    monkeypatch.setattr(live, "_VALIDATED", set())
    if hasattr(getattr(live, "_decode_launch", None), "__wrapped__"):
        monkeypatch.setattr(live, "_decode_launch", live._decode_launch.__wrapped__)
    if hasattr(live, "kernels"):
        monkeypatch.setattr(live.kernels, "_FACE_KERNELS", {})
    return builds


def _up_builds(builds):
    ups = [(name, header, source) for name, header, source in builds if "up" in name]
    assert len(ups) == 1, [name for name, _, _ in builds]
    return ups[0]


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("gpu_class", ["d", "s"])
@pytest.mark.parametrize("rows", [4, 5, 6, 7, 16])
def test_affine_serial_up_route_by_row(rows, gpu_class, dtype, monkeypatch):
    """Six rows keep the two-launch decode; from seven, the serial up streams.

    The two-launch up re-reads its weight for every row and wins the small row
    counts; from seven rows the three-launch up, which streams one affine
    group's scales through all of a row's chunks, is as fast or faster (it
    measured 0.9993 against the parent at seven rows on the M3 Ultra). Both
    activation dtypes' faces stream. The kernels are recorded, not run: both
    routes' numbers are checked against the canonical path and against each
    other by their own tests.
    """
    hidden = 256
    width = HC * hidden
    module = _module(8, False, hidden=hidden, dtype=dtype)
    module.hc_norm.weight = mx.zeros((width,), dtype)
    live = _live()
    monkeypatch.setattr(live, "_gpu_class", lambda: gpu_class)
    builds = _kernel_recorder(live, monkeypatch, rows, width, hidden, LOWRANK, dtype)

    mixed = live.fused_forward(module, mx.ones((1, rows, width), dtype))
    assert mixed is not None
    name, header, source = _up_builds(builds)
    names = [built for built, _, _ in builds]
    if rows <= 6:
        assert "decode_up" in name, names
        assert any("decode_norm_down" in built for built in names), names
        assert "hc_stream_up" not in header + source
    else:
        assert "fused_up" in name, names
        assert any("hc_fused_down" in built for built in names), names
        assert "hc_stream_up" in header, "the streamed up's header is missing"
        assert "hc_stream_up<R>(up, n, a)" in source, "up did not stream the group"


@pytest.mark.parametrize(
    "bits,group,rows,hidden",
    [(4, 64, 7, 256), (8, 32, 7, 256), (8, 64, 6, 5440)],
)
def test_other_faces_and_small_rows_keep_the_generic_up(
    bits, group, rows, hidden, monkeypatch
):
    """Only the stock 8-bit/g64 face at 7-16 rows streams the group.

    A 4-bit or 32-group face and a row count below seven (here forced out of
    the two-launch form by the strip limit) keep the generic serial up.
    """
    width = HC * hidden
    module = _module(bits, False, hidden=hidden, group_size=group)
    module.hc_norm.weight = mx.zeros((width,), module.hc_norm.weight.dtype)
    live = _live()
    monkeypatch.setattr(live, "_gpu_class", lambda: "d")
    builds = _kernel_recorder(
        live, monkeypatch, rows, width, hidden, LOWRANK, module.hc_norm.weight.dtype
    )

    mixed = live.fused_forward(module, mx.ones((1, rows, width), mx.bfloat16))
    assert mixed is not None
    name, header, source = _up_builds(builds)
    assert "fused_up" in name, [built for built, _, _ in builds]
    assert "hc_stream_up" not in header + source


def test_affine_serial_up_streams_an_equal_face_copy(monkeypatch):
    """An equal but distinct face still streams: concurrent cold starts,
    ``affine.face.cache_clear()`` and deep copies hand one out."""
    hidden, dtype = 256, mx.bfloat16
    width = HC * hidden
    module = _module(8, False, hidden=hidden)
    module.hc_norm.weight = mx.zeros((width,), dtype)
    up = module.input_mix_weight_up
    live = _live()
    stock_face = live._decode_face
    copied = replace(stock_face(up, False))
    assert copied == stock_face(up, False) and copied is not stock_face(up, False)
    monkeypatch.setattr(
        live,
        "_decode_face",
        lambda projection, inject: (
            copied if projection is up else stock_face(projection, inject)
        ),
    )
    monkeypatch.setattr(live, "_gpu_class", lambda: "d")
    builds = _kernel_recorder(live, monkeypatch, 7, width, hidden, LOWRANK, dtype)

    mixed = live.fused_forward(module, mx.ones((1, 7, width), dtype))
    assert mixed is not None
    name, header, source = _up_builds(builds)
    assert "fused_up" in name, [built for built, _, _ in builds]
    assert "hc_stream_up" in header, "an equal face's header is missing"
    assert "hc_stream_up<R>(up, n, a)" in source, "an equal face did not stream"


def test_decode_v2_accounts_for_face_table_bytes(monkeypatch):
    """A face with staged tables spends threadgroup memory the form must fit."""
    live = _live()
    monkeypatch.setattr(live, "_gpu_class", lambda: "d")
    hidden_row = (1, HC, 16368, 504, 63, True)
    assert live._decode_v2(*hidden_row, 2)
    assert not live._decode_v2(*hidden_row, 2, table_bytes_down=64)
    hidden_up = (1, HC, 256, 3264, 204, False)
    assert live._decode_v2(*hidden_up, 2)
    assert not live._decode_v2(*hidden_up, 2, table_bytes_up=64)
    # A table that fits exactly is still served; four bytes more are not.
    row_fit = (1, HC, 16320, 504, 63, True)
    assert live._decode_v2(*row_fit, 2, table_bytes_down=96)
    assert not live._decode_v2(*row_fit, 2, table_bytes_down=100)
    up_fit = (1, HC, 256, 3200, 200, False)
    assert live._decode_v2(*up_fit, 2, table_bytes_up=640)
    assert not live._decode_v2(*up_fit, 2, table_bytes_up=644)


def test_decode_launch_accounts_for_face_tables(monkeypatch):
    """A face's staged tables reach the form's bounds through the launch."""
    from omlx.formats import Table

    hidden, rows = 16320, 1
    width = HC * hidden
    module = _module(8, True, hidden=hidden, lowrank=512)
    module.hc_norm.weight = mx.zeros((width,), module.hc_norm.weight.dtype)
    down = module.input_mix_weight_down
    inj = module.block_inject_weight
    live = _live()
    stock_face = live._decode_face

    def with_tables(down_table, inj_table):
        def attach(projection, inject):
            face = stock_face(projection, inject)
            if projection is down:
                return replace(face, tables=(down_table,))
            if projection is inj:
                return replace(face, tables=(inj_table,))
            return face

        return attach

    monkeypatch.setattr(live, "_gpu_class", lambda: "d")
    shared = Table("t", mx.float32, (0.0,) * 24)
    for down_table, inj_table, expected in (
        (
            Table("d", mx.float32, (0.0,) * 12),
            Table("i", mx.float32, (0.0,) * 12),
            "decode_up",
        ),  # 48 + 48 bytes fit
        (
            Table("d", mx.float32, (0.0,) * 12),
            Table("i", mx.float32, (0.0,) * 13),
            "fused_up",
        ),  # 48 + 52 overflow; any dropped face sees 52 at most
        (shared, shared, "decode_up"),  # one deduplicated 96-byte table fits
    ):
        monkeypatch.setattr(live, "_decode_face", with_tables(down_table, inj_table))
        builds = _kernel_recorder(
            live, monkeypatch, rows, width, hidden, 512, module.hc_norm.weight.dtype
        )
        out = live.fused_forward(
            module, mx.ones((1, rows, width), module.hc_norm.weight.dtype)
        )
        assert out is not None
        assert expected in _up_builds(builds)[0], (expected, [b[0] for b in builds])

    hidden, lowrank, rows = 256, 3200, 1
    width = HC * hidden
    module = _module(4, False, hidden=hidden, lowrank=lowrank)
    module.hc_norm.weight = mx.zeros((width,), module.hc_norm.weight.dtype)
    up = module.input_mix_weight_up

    def with_up_table(values):
        table = Table("t", mx.float32, (0.0,) * values)
        return lambda projection, inject: (
            replace(stock_face(projection, inject), tables=(table,))
            if projection is up
            else stock_face(projection, inject)
        )

    for values, expected in ((160, "decode_up"), (161, "fused_up")):
        monkeypatch.setattr(live, "_decode_face", with_up_table(values))
        builds = _kernel_recorder(
            live, monkeypatch, rows, width, hidden, lowrank, module.hc_norm.weight.dtype
        )
        out = live.fused_forward(
            module, mx.ones((1, rows, width), module.hc_norm.weight.dtype)
        )
        assert out is not None
        assert expected in _up_builds(builds)[0], (values, [b[0] for b in builds])


@pytest.mark.parametrize("rows", [1, 7])
def test_concurrent_decode_reads_each_layers_current_weights(rows, monkeypatch):
    live = _live()
    monkeypatch.setattr(live, "_gpu_class", lambda: "d")
    monkeypatch.setattr(live, "_VALIDATED", set())
    mx.random.seed(2670)
    layers = [_module(bits, True, hidden=256) for bits in (8, 8, 4)]
    hyper, pending = _inputs(rows, 256, mx.bfloat16, 2671)
    expected = [oracle.decode(layer, hyper, pending, gpu_class="d") for layer in layers]
    for layer in layers:
        live._eps_array(layer)
    mx.eval(expected)
    together = Barrier(2, timeout=30)

    def check(order):
        for index in order:
            together.wait()
            result = live.fused_forward(layers[index], hyper, pending)
            assert result is not None
            _assert_bitwise(result, expected[index], (rows, index))

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(check, [(0, 2, 1), (1, 0, 2)]))


@pytest.mark.parametrize("bits", [4, 8])
def test_a_low_rank_of_other_chunks_matches_the_oracle(bits, monkeypatch):
    """A low rank of 384: 24 or 48 up chunks, which do not divide the up tile's
    320 threads, so it takes whole chunk rows of them."""
    modules = _modules(bits, 64, mx.bfloat16, seed=2300 + bits, lowrank=384)
    for gpu_class in ("d", "s"):
        _assert_decode(
            monkeypatch,
            modules,
            2560,
            mx.bfloat16,
            (1, 2, 4, 7),
            gpu_class,
            2300,
        )


def test_the_oracle_refuses_what_hc_fused_does_not_take(monkeypatch):
    mx.random.seed(2400)
    module = _module(4, True)
    hyper, (branch, gate) = _inputs(2, 2560, mx.bfloat16, 2400)
    for rows in (0, oracle.MAX_ROWS + 1):
        rows_in, _ = _inputs(rows, 2560, mx.bfloat16, 2400)
        with pytest.raises(ValueError, match="1 to 16 rows"):
            oracle.decode(module, rows_in, gpu_class="d")
    for write in (
        (branch[..., :-64], gate),
        (branch.reshape(2, 1, 2560), gate),
        (branch, gate[..., :-1]),
        (branch, gate.reshape(2, 1, HC)),
        (branch.astype(mx.float16), gate),
        (branch, gate.astype(mx.float16)),
    ):
        with pytest.raises(ValueError, match="no such pending write"):
            oracle.decode(module, hyper, write, gpu_class="d")
    prefill, (branch, gate) = _inputs(17, 2560, mx.bfloat16, 2401)
    with pytest.raises(ValueError, match="no such pending write"):
        oracle.prefill(module, prefill, (branch, gate.reshape(17, 1, HC)))
    # 5440: 22 norm strips a stream, 21 row groups from 3 rows.
    wide = _module(4, True, hidden=5440)
    rows_in, _ = _inputs(3, 5440, mx.bfloat16, 2401)
    with pytest.raises(ValueError, match="do not take this call"):
        oracle.decode(wide, rows_in, gpu_class="d", two_launch=True)
    expected = oracle.decode(wide, rows_in, gpu_class="d", two_launch=False)
    live, automatic = _decode(monkeypatch, wide, rows_in, None, "d")
    _assert_bitwise(automatic, expected, "oracle fallback")
    _assert_bitwise(live, expected, "live fallback")


@pytest.mark.parametrize("hidden,lowrank", [(8192, 320), (256, 1792)])
def test_fp32_decode_respects_each_threadgroup_memory_limit(
    hidden, lowrank, monkeypatch
):
    """The stream or the up tile fits in 32 KiB at two bytes per activation.
    Neither fits at four. Refuse the forced launch before it reaches Metal."""
    mx.random.seed(2402)
    module = _module(8, True, hidden=hidden, dtype=mx.float32, lowrank=lowrank)
    rows_in, _ = _inputs(1, hidden, mx.float32, 2402)
    assert oracle.two_launch_fits(1, HC, hidden, lowrank, 8, True, "d")

    def oversized(*args):
        pytest.fail("dispatched a two-launch tile exceeding threadgroup memory")

    monkeypatch.setattr(oracle, "_norm_down", oversized)
    monkeypatch.setattr(_live(), "_kernel_norm_down", oversized)
    with pytest.raises(ValueError, match="do not take this call"):
        oracle.decode(module, rows_in, gpu_class="d", two_launch=True)
    expected = oracle.decode(module, rows_in, gpu_class="d", two_launch=False)
    live, automatic = _decode(monkeypatch, module, rows_in, None, "d")
    _assert_bitwise(automatic, expected, "FP32 oracle fallback")
    _assert_bitwise(live, expected, "FP32 live fallback")
