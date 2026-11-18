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

import mlx.core as mx
import pytest

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


def _projection():
    from mlx_vlm.models.qwen4_exp import hc_projection

    return hc_projection


# Each source the oracle runs, by the live module and name it is kept under.
_SOURCES = {
    "HEADER": ("projection", "_HEADER"),
    "TG_HEADER": ("fused", "_TG_HEADER"),
    "N_SOURCE": ("fused", "_N_SOURCE"),
    "NP_SOURCE": ("fused", "_NP_SOURCE"),
    "WN_SOURCE": ("fused", "_WN_SOURCE"),
    "WND_SOURCE": ("fused", "_WND_SOURCE"),
    "D_SOURCE": ("fused", "_D_SOURCE"),
    "U_SOURCE": ("fused", "_U_SOURCE"),
    "NDN_SOURCE": ("fused", "_NDN_SOURCE"),
    "NDNW_SOURCE": ("fused", "_NDNW_SOURCE"),
    "U2_SOURCE": ("fused", "_U2_SOURCE"),
    "TI_SOURCE": ("fused", "_TI_SOURCE"),
    "V2_HEADER": ("fused", "_V2_HEADER"),
}


@pytest.mark.parametrize("name", sorted(_SOURCES))
def test_oracle_sources_are_the_live_ones(name):
    module, live = _SOURCES[name]
    source = getattr(_live() if module == "fused" else _projection(), live)
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
    assert set(_SOURCES) - derived | pieces <= literals
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


def _modules(bits, group_size, dtype, hidden=2560, seed=0):
    """A module with the inject rows and one without, the same down and up."""
    mx.random.seed(seed)
    with_inject = _module(bits, True, hidden=hidden, group_size=group_size, dtype=dtype)
    without = _module(bits, False, hidden=hidden, group_size=group_size, dtype=dtype)
    without.hc_norm = with_inject.hc_norm
    without.input_mix_weight_down = with_inject.input_mix_weight_down
    without.input_mix_weight_up = with_inject.input_mix_weight_up
    return with_inject, without


def _decode(monkeypatch, module, hyper, write, gpu_class):
    hc_fused = _live()
    monkeypatch.setattr(hc_fused, "_gpu_class", lambda: gpu_class)
    live = hc_fused.fused_forward(module, hyper, write)
    assert live is not None
    return live, oracle.decode(module, hyper, write, gpu_class=gpu_class)


def _assert_fits_fp32(module, hidden, rows, gpu_class):
    """two_launch_fits counts 2-byte rows, as hc_fused does: an FP32 shape of
    these tests fits the two-launch kernels' threadgroup memory at 4 bytes."""
    lowrank = module.hc_lowrank
    chunks = oracle.up_chunks(module["input_mix_weight_up"].bits, lowrank)
    outputs = oracle.up_outputs(rows, gpu_class)
    assert hidden * 4 + 32 <= oracle.TG_MEMORY_BYTES
    assert lowrank * 4 + outputs * (chunks + 1) * 4 <= oracle.TG_MEMORY_BYTES


def _assert_decode(
    monkeypatch, modules, hidden, dtype, rows_list, gpu_class, seed, writes=True
):
    for module in modules:
        for rows in rows_list:
            if rows > oracle.V2_MAX_ROWS and gpu_class != "d":
                continue  # the three-launch kernels have one geometry
            if dtype == mx.float32:
                _assert_fits_fp32(module, hidden, rows, gpu_class)
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
                for gpu_class in ("d", "s"):
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
    for module in _modules(bits, group_size, dtype, seed=seed):
        inject = "block_inject_weight" in module
        for gpu_class in ("d", "s"):
            for rows in range(1, oracle.MAX_ROWS + 1):
                assert oracle.two_launch_fits(
                    rows, HC, 2560, LOWRANK, bits, inject, gpu_class
                )
                hyper, pending = _inputs(rows, 2560, dtype, seed + rows)
                for write in (None, pending):
                    _assert_bitwise(
                        oracle.decode(
                            module, hyper, write, gpu_class=gpu_class, two_launch=True
                        ),
                        oracle.decode(
                            module, hyper, write, gpu_class=gpu_class, two_launch=False
                        ),
                        ("forms", gpu_class, rows, inject, write is None),
                    )


# Shapes on each side of every clause of the two-launch fit: three streams;
# a low rank of whole row groups or not (328); the norm's strips against the
# row groups (5376, 5440); the stream in threadgroup memory (16320, 16384, and
# 16368 exactly at the limit); the up tile's threadgroup memory (512 at 8 bits
# over it; 960, over it only with its activations and the staging's padding
# column; 3264 at 4 bits and 32 outputs, exactly at the limit; 2560).
_FORM_SHAPES = [
    (hc, hidden, lowrank)
    for hc in (3, 4)
    for hidden in (2560, 5376, 5440, 16320, 16368, 16384)
    for lowrank in (320, 328, 384, 512, 960, 2560, 3264)
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
                        live = hc_fused._decode_v2(
                            rows, hc, hidden, lowrank, bits, inject
                        )
                        fits = oracle.two_launch_fits(
                            rows, hc, hidden, lowrank, bits, inject, gpu_class
                        )
                        seen.add((rows <= oracle.V2_MAX_ROWS, fits))
                        assert live == oracle.takes_two_launches(
                            rows, hc, hidden, lowrank, bits, inject, gpu_class
                        ), (gpu_class, hc, hidden, lowrank, rows, bits, inject)
    assert seen == {(True, True), (True, False), (False, True), (False, False)}


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
            chunks = hc_fused._up_chunks(bits, lowrank)
            assert oracle.up_chunks(bits, lowrank) == chunks
            # hc_fused's _kernel_up_mix: whole chunk rows within its threads.
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
                        for gpu_class in ("d", "s"):
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


def _low_rank(module, lowrank, bits, group_size=64):
    """``module`` with down and up projections of another low rank."""
    import mlx.nn as nn

    width = module.hc_count * module.hidden_size
    dtype = module.hc_norm.weight.dtype
    module.hc_lowrank = lowrank
    for name, (k, n) in (
        ("input_mix_weight_down", (width, lowrank)),
        ("input_mix_weight_up", (lowrank, width)),
    ):
        layer = nn.QuantizedLinear(k, n, bias=False, group_size=group_size, bits=bits)
        scales = mx.abs(mx.random.normal(layer.scales.shape)) * 0.01 + 0.002
        layer.scales = scales.astype(dtype)
        layer.biases = (mx.random.normal(layer.biases.shape) * 0.005).astype(dtype)
        setattr(module, name, layer)
    mx.eval(module.parameters())
    return module


@pytest.mark.parametrize("bits", [4, 8])
def test_a_low_rank_of_other_chunks_matches_the_oracle(bits, monkeypatch):
    """A low rank of 384: 24 or 48 up chunks, which do not divide the up tile's
    320 threads, so it takes whole chunk rows of them."""
    mx.random.seed(2300 + bits)
    module = _low_rank(_module(bits, True), 384, bits)
    without = _module(bits, False)
    without.hc_norm = module.hc_norm
    without.hc_lowrank = 384
    without.input_mix_weight_down = module.input_mix_weight_down
    without.input_mix_weight_up = module.input_mix_weight_up
    for gpu_class in ("d", "s"):
        _assert_decode(
            monkeypatch,
            [module, without],
            2560,
            mx.bfloat16,
            (1, 2, 4, 7),
            gpu_class,
            2300,
        )


def test_the_oracle_refuses_what_hc_fused_does_not_take():
    mx.random.seed(2400)
    module = _module(4, True)
    hyper, (branch, gate) = _inputs(2, 2560, mx.bfloat16, 2400)
    for rows in (0, oracle.MAX_ROWS + 1):
        rows_in, _ = _inputs(rows, 2560, mx.bfloat16, 2400)
        with pytest.raises(ValueError, match="1 to 16 rows"):
            oracle.decode(module, rows_in, gpu_class="d")
    for write in ((branch[..., :-64], gate), (branch, gate.astype(mx.float16))):
        with pytest.raises(ValueError, match="no such pending write"):
            oracle.decode(module, hyper, write, gpu_class="d")
    # 5440: 22 norm strips a stream, 21 row groups from 3 rows.
    wide = _module(4, True, hidden=5440)
    rows_in, _ = _inputs(3, 5440, mx.bfloat16, 2401)
    with pytest.raises(ValueError, match="do not take this call"):
        oracle.decode(wide, rows_in, gpu_class="d", two_launch=True)
