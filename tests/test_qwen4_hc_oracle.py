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
(hc_fused takes FP32 rows in its decode, including same-dtype pending
writes; the oracle checks below run write-free FP32 rows).
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

from omlx.formats import Table, affine, kernels
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from tests.oracles import hc_fused as oracle
from tests.test_qwen4_hc_fused import HC, LOWRANK, _module, _reference_fp32, _rel_l2

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
    "N_SOURCE": ("fused", "_N_SOURCE"),
    "NP_SOURCE": ("fused", "_NP_SOURCE"),
    "WN_SOURCE": ("fused", "_WN_SOURCE"),
    "WND_SOURCE": ("fused", "_WND_SOURCE"),
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
    derived = {
        "NP_SOURCE",
        "WND_SOURCE",
        "NDN_SOURCE",
        "NDNW_SOURCE",
        "V2_HEADER",
    }
    pieces = {"NDN_SOURCE_TEMPLATE", "NDN_LOAD", "NDN_WRITE_LOAD"}
    frozen = {
        "D_SOURCE",
        "U_SOURCE",
        "U2_SOURCE",
        "TI_SOURCE",
        "TG_HEADER",
    }
    assert (set(_SOURCES) - derived) | pieces | frozen <= literals
    live_tree = ast.parse(inspect.getsource(_live()))
    live_literals = {
        node.targets[0].id
        for node in live_tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    assert "_TAIL_INJECT" in live_literals
    assert "_TI_SOURCE" not in live_literals and "_TG_HEADER" not in live_literals
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


def _assert_prefill(live, expected, module, hyper, write, what):
    """The tail inject and a written stream are bitwise; the mixed path runs
    the layers' own route (a reordered accumulation), judged by the C2 bar
    against an FP32 reference (rule 33, class 1)."""
    live = live if isinstance(live, tuple) else (live,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    assert len(live) == len(expected), what
    mx.eval(live, expected)
    if len(live) == 3:
        mixed, middle, injection = live
        expected_mixed, expected_middle, expected_injection = expected
        if write is not None:
            assert mx.array_equal(
                middle.view(_view(middle.dtype)),
                expected_middle.view(_view(expected_middle.dtype)),
            ).item(), what
            source = middle
        else:
            source = hyper
    elif len(live) == 2:
        mixed, middle = live
        mixed_expected, expected_middle = expected
        if write is not None:
            assert mx.array_equal(
                middle.view(_view(middle.dtype)),
                expected_middle.view(_view(expected_middle.dtype)),
            ).item(), what
            source = middle
        else:
            source = hyper
        injection = expected_injection = None
        expected_mixed = mixed_expected
    else:
        (mixed,) = live
        (expected_mixed,) = expected
        injection = expected_injection = None
        source = hyper
    # Rule 33, class 1: the mixed path (and a fallback injection, which runs
    # the same dense route) is a reordered accumulation, judged by the C2 bar
    # against an FP32 reference from the same dequantized weights, not
    # bitwise against the MLX quantized matmul it replaces. Where the tail
    # inject's face tiles the width, its injection stays bitwise.
    reference, reference_injection = _reference_fp32(module, source)
    rel_new = _rel_l2(mixed, reference)
    rel_base = _rel_l2(expected_mixed, reference)
    assert rel_new <= 1.05 * rel_base + 1e-6, (what, rel_new, rel_base)
    if injection is not None:
        inject = module.block_inject_weight
        tiled = (HC * module.hidden_size) % (256 * oracle.pack_factor(inject.bits)) == 0
        if tiled:
            assert mx.array_equal(
                injection.view(_view(injection.dtype)),
                expected_injection.view(_view(expected_injection.dtype)),
            ).item(), what
        else:
            rel_new = _rel_l2(injection, reference_injection)
            rel_base = _rel_l2(expected_injection, reference_injection)
            assert rel_new <= 1.05 * rel_base + 1e-6, (
                what,
                "inject",
                rel_new,
                rel_base,
            )


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
    hc_fused.rebuild_plan(module)
    live = hc_fused.fused_forward(module, hyper, write)
    assert live is not None
    return live, oracle.decode(module, hyper, write, gpu_class=gpu_class)


@pytest.mark.parametrize("rows", [2, 7])
def test_warm_decode_does_not_rerecord_face_kernel(rows, monkeypatch):
    """OPT-1: warm decode calls prebuilt native launches directly without
    re-entering _FaceKernel._record on either two-launch or three-launch rows."""
    live = _live()
    monkeypatch.setattr(live.kernels, "_FACE_KERNELS", {})
    monkeypatch.setattr(live, "_VALIDATED", set())
    if hasattr(getattr(live, "_decode_launch", None), "__wrapped__"):
        monkeypatch.setattr(live, "_decode_launch", live._decode_launch.__wrapped__)
    seen = []
    real_face_kernel = live.kernels.face_kernel

    def capture(*args, **kwargs):
        kernel = real_face_kernel(*args, **kwargs)
        seen.append(kernel)
        return kernel

    monkeypatch.setattr(live.kernels, "face_kernel", capture)
    module = _module(8, True, hidden=256)
    hyper, pending = _inputs(rows, 256, mx.bfloat16, 2570 + rows)
    first = live.fused_forward(module, hyper, pending)
    assert first is not None
    mx.eval(first)
    assert seen
    records = []
    for kernel in seen:
        real_record = kernel._record
        monkeypatch.setattr(
            kernel,
            "_record",
            lambda *a, _orig=real_record, **kw: records.append(1) or _orig(*a, **kw),
        )
    second = live.fused_forward(module, hyper, pending)
    assert second is not None
    mx.eval(second)
    assert records == [], f"warm decode re-recorded _FaceKernel {len(records)} times"
    _assert_bitwise(second, first, ("warm decode opt-1", rows))


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
        rps = original(rows)
        if rows == 4:
            seen.append(rps)
        return rps

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
    # The oracle parity checks write-free FP32 rows; FP32 pending writes are
    # covered by test_fused_forward_supports_fp32_pending_write.
    writes = dtype != mx.float32
    _assert_decode(monkeypatch, modules, hidden, dtype, rows, gpu_class, seed, writes)


@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_float32_decode_matches_the_oracle(bits, group_size, monkeypatch):
    seed = 1700 + 10 * bits + group_size
    modules = _modules(bits, group_size, mx.float32, seed=seed)
    rows = range(1, oracle.MAX_ROWS + 1)
    for gpu_class in ("d", "s"):
        # The oracle parity checks write-free FP32 rows; FP32 pending writes
        # are covered by test_fused_forward_supports_fp32_pending_write.
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


def test_tail_inject_ok_checks_vpt_and_table_bytes():
    """The tail inject's face must tile 256 threads and fit the threadgroup."""
    live = _live()
    dtype = mx.bfloat16
    qmv8 = affine.face(8, 64, False, dtype)
    assert qmv8.vpt == 4
    assert live._tail_inject_ok(10240, 2560, 4, dtype, qmv8)
    fast4 = affine.face(4, 64, True, dtype)
    assert fast4.vpt == 16
    assert not live._tail_inject_ok(10240, 2560, 4, dtype, fast4)
    empty = replace(qmv8, vpt=0)
    assert not live._tail_inject_ok(10240, 2560, 4, dtype, empty)
    unit = replace(qmv8, vpt=1)
    assert live._tail_inject_ok(10240, 2560, 4, dtype, unit)
    assert not live._tail_inject_ok(4608, 1152, 4, dtype, qmv8)
    assert not live._tail_inject_ok(0, 0, 4, dtype, qmv8)
    assert not live._tail_inject_ok(10240, -2560, -4, dtype, qmv8)
    assert not live._tail_inject_ok(10240, 2560, 0, dtype, qmv8)
    assert not live._tail_inject_ok(10240, 2560, 4, dtype, None)
    # width > 0, the width == hc * hidden upper bound, and the dtype's byte
    # size are each load-bearing (pre-hc-8b9f3528 LOW-1).
    assert not live._tail_inject_ok(0, 2560, 0, dtype, qmv8)
    assert not live._tail_inject_ok(-10240, 2560, -4, dtype, qmv8)
    assert not live._tail_inject_ok(2048, 2560, 4, dtype, qmv8)
    assert not live._tail_inject_ok(32768, 8192, 4, mx.float32, qmv8)
    assert not live._tail_inject_ok(2560, 2560, 4, dtype, qmv8)
    assert not live._tail_inject_ok(3072, 1024, 3, dtype, qmv8)
    assert not live._tail_inject_ok(4096, 256, 16, dtype, qmv8)
    # H = 16128 keeps the width whole 256-thread groups and leaves 384
    # bytes of the 32 KiB threadgroup for tables after the stream and
    # partials.
    tabled = replace(qmv8, tables=(Table("t", mx.float32, (0.0,) * 96),))
    assert live._tail_inject_ok(64512, 16128, 4, dtype, tabled)
    over = replace(qmv8, tables=(Table("t", mx.float32, (0.0,) * 97),))
    assert not live._tail_inject_ok(64512, 16128, 4, dtype, over)


def test_prefill_tail_inject_uses_dot_face_and_finish():
    """The tail inject reads the inject face: its finish scales the row's sum."""
    from tests.test_qwen4_hc_fused import _hc_toy_format, _plugin_projection

    hidden, dtype, rows = 2560, mx.bfloat16, 17
    width = HC * hidden
    mx.random.seed(2678)
    module = _module(8, True, hidden=hidden)
    # The staged stream is the normed row, not sigmoid(up) * normed: drive the
    # up projection's gate to zero so the assertion would fail if it were.
    module.input_mix_weight_up.biases = mx.full(
        module.input_mix_weight_up.biases.shape, -10.0, dtype
    )
    # A plugin inject: its own codes/name (D11) and a non-identity finish.
    inject = _plugin_projection(_hc_toy_format(8), width, HC, 90)
    module.block_inject_weight = inject
    codes = mx.full((HC, width), 128, mx.uint8)
    codes[:, 0] = 129
    codes[:, hidden] = 129
    inject.qcodes = codes
    base_gains = mx.array([1.0, 1.5, 2.0, 2.5, 0.0, 0.0, 0.0, 0.0], mx.float32)
    hyper, _ = _inputs(rows, hidden, dtype, 2679)
    live = _live()

    def injection_of(gains):
        inject.gains = gains
        live.rebuild_plan(module)
        out = live.prefill_forward(module, hyper, None, nax=False)
        assert out is not None
        mx.eval(out)
        return out[-1]

    base = injection_of(base_gains)
    doubled = injection_of(2.0 * base_gains)
    # The dot's value: each code adds its own stream's normed element.
    flat = hyper.reshape(rows, width)
    normed = live._kernel_norm(
        module.hc_norm.weight,
        live._eps_array(module),
        flat,
        rows,
        HC,
        hidden,
        dtype,
        precise=True,
    )
    normed = normed.reshape(rows, width)
    total = normed[:, 0].astype(mx.float32) + normed[:, hidden].astype(mx.float32)
    for t in range(HC):
        expected = 2.0 * mx.sigmoid(float(base_gains[t]) * total / float(HC))
        gap = mx.abs(base[0, :, t].astype(mx.float32) - expected)
        assert float(gap.max()) < 0.02, "the face's dot is not the staged stream's"
    # finish scales the dot's FP32 sum before the sigmoid: logit(inj/2) doubles.
    q_base = mx.log(base / (2.0 - base))
    q_doubled = mx.log(doubled / (2.0 - doubled))
    for t in range(4):
        assert float(mx.abs(q_doubled[0, :, t] - 2.0 * q_base[0, :, t]).max()) < 0.12


@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_float32_tail_inject_matches_oracle_bits(bits, group_size):
    """The FP32 sums, where a 16-bit output would mask a reordering."""
    rows = 17
    mx.random.seed(2800 + bits + group_size)
    module = _module(bits, True, group_size=group_size, dtype=mx.float32)
    inject = module.block_inject_weight
    hidden = 2560
    width = HC * hidden
    up = (mx.random.normal((rows, width)) * 3).astype(mx.float32)
    normed = (mx.random.normal((rows, width)) * 0.05).astype(mx.float32)
    live = _live()
    face = affine.face(bits, group_size, False, mx.float32)
    launch = live._tail_inject_launch(HC, hidden, mx.float32, face)
    assert launch is not None
    actual = launch(rows, up, normed, (inject.weight, inject.scales, inject.biases))
    expected = oracle._kernel(
        "ti_f32",
        ["up", "xn", "inject_w", "inject_s", "inject_b"],
        ["mixed", "inj"],
        oracle.TI_SOURCE,
        header=oracle.V2_HEADER,
    )(
        inputs=[up, normed, inject.weight, inject.scales, inject.biases],
        template=[
            ("T", mx.float32),
            ("BITS_I", bits),
            ("GS_I", group_size),
            ("K", width),
            ("H", hidden),
            ("HC", HC),
        ],
        grid=(256, 1, rows),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, hidden), (rows, HC)],
        output_dtypes=[mx.float32, mx.float32],
    )
    _assert_bitwise(
        tuple(actual), tuple(expected), ("tail inject fp32", bits, group_size)
    )


def test_prefill_tail_inject_plan_refreshes_and_records_instance(monkeypatch):
    """A swapped layer or a new dtype rebuilds the immutable plan on rebuild_plan."""
    hidden, dtype, rows = 2560, mx.bfloat16, 17
    mx.random.seed(2810)
    module = _module(8, True, hidden=hidden)
    hyper, pending = _inputs(rows, hidden, dtype, 2811)
    live = _live()
    seen = []
    real_face_kernel = live.kernels.face_kernel

    def record(*args, **kwargs):
        kernel = real_face_kernel(*args, **kwargs)
        seen.append(kernel)
        return kernel

    monkeypatch.setattr(live.kernels, "face_kernel", record)
    monkeypatch.setattr(live.kernels, "_FACE_KERNELS", {})
    out = live.prefill_forward(module, hyper, pending, nax=False)
    assert out is not None
    assert seen and len(seen[-1].instances()) == 1
    _assert_bitwise(out, oracle.prefill(module, hyper, pending), "first plan")
    first = module.__dict__["_omlx_hc_plan"]
    face_calls = []
    records = []
    real_face = live._projection_face
    real_record = seen[-1]._record

    def counted_face(*args):
        face_calls.append(1)
        return real_face(*args)

    def counted_record(*args, **kwargs):
        records.append(1)
        return real_record(*args, **kwargs)

    monkeypatch.setattr(live, "_projection_face", counted_face)
    monkeypatch.setattr(seen[-1], "_record", counted_record)
    again = live.prefill_forward(module, hyper, pending, nax=False)
    assert again is not None
    assert module.__dict__["_omlx_hc_plan"] is first
    assert not face_calls and not records, (len(face_calls), len(records))
    other = _module(8, True, hidden=hidden, group_size=32)
    module.block_inject_weight = other.block_inject_weight
    live.rebuild_plan(module)
    out = live.prefill_forward(module, hyper, pending, nax=False)
    assert out is not None
    plan = module.__dict__["_omlx_hc_plan"]
    assert plan.inject is module.block_inject_weight and plan.inject is not first.inject
    _assert_bitwise(out, oracle.prefill(module, hyper, pending), "swapped layer")
    module.set_dtype(mx.float16)
    live.rebuild_plan(module)
    half = hyper.astype(mx.float16)
    half_pending = (
        pending[0].astype(mx.float16),
        pending[1].astype(mx.float16),
    )
    out = live.prefill_forward(module, half, half_pending, nax=False)
    assert out is not None
    plan = module.__dict__["_omlx_hc_plan"]
    assert plan.dtype == mx.float16
    _assert_bitwise(out, oracle.prefill(module, half, half_pending), "new input dtype")


def test_concurrent_prefill_tail_inject_matches_oracle(monkeypatch):
    """Two threads on different formats each get their own layer's bits."""
    hidden, rows = 2560, 17
    modules = {bits: _module(bits, True, hidden=hidden) for bits in (4, 8)}
    inputs = {bits: _inputs(rows, hidden, mx.bfloat16, 2820 + bits) for bits in modules}
    live = _live()
    barrier = Barrier(2)

    def run(bits):
        module = modules[bits]
        hyper, pending = inputs[bits]
        barrier.wait()
        first = None
        for _ in range(3):
            out = live.prefill_forward(module, hyper, pending, nax=False)
            assert out is not None
            mx.eval(out)
            if first is None:
                first = out
                _assert_prefill(
                    out,
                    oracle.prefill(module, hyper, pending),
                    module,
                    hyper,
                    pending,
                    ("thread", bits),
                )
            else:
                # A thread's own repeat must be the same bits: a shared
                # specialization or record racing would show here.
                for a, b in zip(out, first):
                    assert mx.array_equal(
                        a.view(_view(a.dtype)), b.view(_view(b.dtype))
                    ).item(), ("thread", bits)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, bits) for bits in modules]
        for future in futures:
            future.result()

    # A cold shared face: two threads race the first record of one face kernel.
    with monkeypatch.context() as inner:
        inner.setattr(live.kernels, "_FACE_KERNELS", {})
        inner.setattr(live, "_VALIDATED", set())
        cold = [_module(8, True, hidden=hidden) for _ in range(2)]
        cold_inputs = [_inputs(rows, hidden, mx.bfloat16, 2830 + i) for i in range(2)]
        cold_barrier = Barrier(2)

        def cold_run(index):
            module, (hyper, pending) = cold[index], cold_inputs[index]
            cold_barrier.wait()
            out = live.prefill_forward(module, hyper, pending, nax=False)
            assert out is not None
            _assert_prefill(
                out,
                oracle.prefill(module, hyper, pending),
                module,
                hyper,
                pending,
                ("cold", index),
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(cold_run, i) for i in range(2)]
            for future in futures:
                future.result()
        assert len(live.kernels.instances()) == 1, live.kernels.instances()
        cold_sigs = {
            sig
            for sig in live._VALIDATED
            if isinstance(sig, tuple) and sig and sig[0] == "prefill"
        }
        assert len(cold_sigs) == 1, cold_sigs


def test_plan_refresh_does_not_orphan_tail_inject_face_kernels(monkeypatch):
    """Restoring _FACE_KERNELS leaves no orphaned tail-inject launch behind."""
    live = _live()
    monkeypatch.setattr(live.kernels, "_FACE_KERNELS", {})
    with monkeypatch.context() as inner:
        test_prefill_tail_inject_plan_refreshes_and_records_instance(inner)
    assert live.kernels.instances() == ()
    hidden, rows = 2560, 17
    module = _module(8, True, hidden=hidden, group_size=32)
    hyper, pending = _inputs(rows, hidden, mx.bfloat16, 2840)
    module.set_dtype(mx.float16)
    half, half_pending = hyper.astype(mx.float16), (
        pending[0].astype(mx.float16),
        pending[1].astype(mx.float16),
    )
    out = live.prefill_forward(module, half, half_pending, nax=False)
    assert out is not None
    recorded = live.kernels.instances()
    assert len(recorded) == 1 and recorded[0].outputs == (
        mx.float16,
        mx.float16,
    ), recorded


def test_prefill_validates_distinct_faces_once_across_row_counts(monkeypatch):
    """One validation per tail-inject specialization, never per row count."""
    hidden, dtype = 2560, mx.bfloat16
    mx.random.seed(2671)
    module = _module(8, True, hidden=hidden)
    module.hc_norm.weight = mx.zeros((HC * hidden,), dtype)
    inject_layer = module.block_inject_weight
    live = _live()
    monkeypatch.setattr(live, "_VALIDATED", set())
    if hasattr(getattr(live, "_decode_launch", None), "__wrapped__"):
        monkeypatch.setattr(live, "_decode_launch", live._decode_launch.__wrapped__)
    monkeypatch.setattr(live, "_KERNELS", {})
    monkeypatch.setattr(live.kernels, "_FACE_KERNELS", {})

    def prefill_signatures():
        return {
            signature
            for signature in live._VALIDATED
            if isinstance(signature, tuple) and signature and signature[0] == "prefill"
        }

    hyper17, pending17 = _inputs(17, hidden, dtype, 2672)
    live.prefill_forward(module, hyper17, pending17, nax=False)
    first = prefill_signatures()
    assert len(first) == 1
    assert not any(callable(x) for sig in first for x in sig), first
    hyper33, pending33 = _inputs(33, hidden, dtype, 2673)
    live.prefill_forward(module, hyper33, pending33, nax=False)
    assert prefill_signatures() == first, "the row count must not re-validate"
    same = _module(8, True, hidden=hidden)
    same.hc_norm.weight = mx.zeros((HC * hidden,), dtype)
    live.prefill_forward(same, hyper33, pending33, nax=False)
    assert (
        prefill_signatures() == first
    ), "a second module with one face must not re-validate"
    live.prefill_forward(same, hyper33, None, nax=False)
    assert len(prefill_signatures()) == 2, "a write-free call validates its own"
    other = _module(8, True, hidden=hidden, group_size=32)
    other.hc_norm.weight = mx.zeros((HC * hidden,), dtype)
    hyper65, pending65 = _inputs(65, hidden, dtype, 2674)
    live.prefill_forward(other, hyper65, pending65, nax=False)
    assert len(prefill_signatures()) == 3, "a distinct face validates its own"
    assert module.__dict__["_omlx_hc_plan"].inject is inject_layer
    fallback = _module(8, True, hidden=1152)
    fallback.hc_norm.weight = mx.zeros((HC * 1152,), dtype)
    fb_hyper, fb_pending = _inputs(17, 1152, dtype, 2675)
    live.prefill_forward(fallback, fb_hyper, fb_pending, nax=False)
    fb_plan = fallback.__dict__["_omlx_hc_plan"]
    assert (fb_plan.tail_launch, fb_plan.tail_face) == (None, None)
    assert ("prefill", dtype, HC, 1152, True, None, ()) in prefill_signatures()


def test_tail_inject_ok_mirrors_the_kernel_s_hc_assert():
    """The eligibility's hc law matches _TAIL_INJECT's static_asserts.

    They require HC * VT == 256 with VT = H / PER = 256 / HC and VT % 32
    == 0, so HC must divide 8; every other hc fails to compile.
    """
    live = _live()
    dtype = mx.bfloat16
    face = affine.face(8, 64, False, dtype)
    hidden = 1024
    for hc in range(1, 33):
        width = hc * hidden
        assert live._tail_inject_ok(width, hidden, hc, dtype, face) == (8 % hc == 0), hc


def test_hc_fused_drops_legacy_tail_headers():
    """The tail inject runs over the face: the hand-written sources are gone."""
    live = _live()
    assert not hasattr(live, "_TI_SOURCE")
    assert not hasattr(live, "_TG_HEADER")
    assert kernels.hand_tables(live._TAIL_INJECT) == ()


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
                _assert_prefill(
                    live,
                    oracle.prefill(module, hyper, write),
                    module,
                    hyper,
                    write,
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
                        _assert_prefill(
                            live,
                            oracle.prefill(m, rows_in, write),
                            m,
                            rows_in,
                            write,
                            what,
                        )
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
            _assert_prefill(
                live,
                oracle.prefill(module, hyper, write),
                module,
                hyper,
                write,
                ("prefill", rows),
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
    stock_face = live._projection_face
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
        "_projection_face",
        lambda p, k, n, dtype, inject: (
            toy if p is layer else stock_face(p, k, n, dtype, inject)
        ),
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
    stock_face = live._projection_face
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
        "_projection_face",
        lambda p, k, n, dtype, inject: (
            toy if p is up else stock_face(p, k, n, dtype, inject)
        ),
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
    live.rebuild_plan(module)
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
        entry = (name, spec.get("header", ""), spec.get("source", ""))

        def invoke(**kwargs):
            builds.append(entry)
            if "fused_write_norm" in name:
                return [
                    mx.zeros((rows, width), dtype),
                    mx.zeros((rows, width), dtype),
                ]
            if "fused_norm" in name:
                return [mx.zeros((rows, width), dtype)]
            if "hc_fused_down" in name:
                return [
                    mx.zeros((rows, lowrank), dtype),
                    mx.zeros((rows, HC), dtype),
                ]
            if "decode_write_norm_down" in name:
                return [
                    mx.zeros((rows, width), dtype),
                    mx.zeros((rows, width), dtype),
                    mx.zeros((rows, HC, lowrank + HC), mx.float32),
                ]
            if "decode_norm_down" in name:
                return [
                    mx.zeros((rows, width), dtype),
                    mx.zeros((rows, HC, lowrank + HC), mx.float32),
                ]
            return [
                mx.zeros((rows, hidden), dtype),
                mx.zeros((rows, HC), dtype),
            ]

        return invoke

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
    stock_face = live._projection_face
    copied = replace(stock_face(up, LOWRANK, width, dtype, False))
    assert copied == stock_face(up, LOWRANK, width, dtype, False)
    assert copied is not stock_face(up, LOWRANK, width, dtype, False)
    monkeypatch.setattr(
        live,
        "_projection_face",
        lambda projection, k, n, face_dtype, inject: (
            copied
            if projection is up
            else stock_face(projection, k, n, face_dtype, inject)
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
    hidden, rows = 16320, 1
    width = HC * hidden
    live = _live()
    stock_face = live._projection_face
    monkeypatch.setattr(live, "_gpu_class", lambda: "d")
    up_table = Table("u", mx.float32, (0.0,) * 5)  # 20 B, outside both bounds

    def case(down_table, inj_table, lowrank=512, inject=True):
        # A fresh module per case resolves the patched faces on its first call.
        module = _module(8, inject, hidden=hidden, lowrank=lowrank)
        module.hc_norm.weight = mx.zeros((width,), module.hc_norm.weight.dtype)
        down = module.input_mix_weight_down
        inj = module.block_inject_weight if inject else None

        def attach(projection, k, n, dtype, inject_flag):
            face = stock_face(projection, k, n, dtype, inject_flag)
            if projection is down:
                return replace(face, tables=(down_table,))
            if inj is not None and projection is inj:
                return replace(face, tables=(inj_table,))
            return replace(face, tables=(up_table,))

        monkeypatch.setattr(live, "_projection_face", attach)
        return module

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
        module = case(down_table, inj_table)
        builds = _kernel_recorder(
            live, monkeypatch, rows, width, hidden, 512, module.hc_norm.weight.dtype
        )
        out = live.fused_forward(
            module, mx.ones((1, rows, width), module.hc_norm.weight.dtype)
        )
        assert out is not None
        assert expected in _up_builds(builds)[0], (expected, [b[0] for b in builds])

    # Without inject rows the down face's own table still counts.
    plain_table = Table("d", mx.float32, (0.0,) * 25)  # 100 B, over the bound
    plain = case(plain_table, None, inject=False)
    builds = _kernel_recorder(
        live, monkeypatch, rows, width, hidden, 512, plain.hc_norm.weight.dtype
    )
    out = live.fused_forward(
        plain, mx.ones((1, rows, width), plain.hc_norm.weight.dtype)
    )
    assert out is not None
    assert (
        "fused_up" in _up_builds(builds)[0]
    ), "the down table must count without inject"

    hidden, lowrank, rows = 256, 3200, 1
    width = HC * hidden
    for values, expected in ((160, "decode_up"), (161, "fused_up")):
        table = Table("t", mx.float32, (0.0,) * values)
        module = _module(4, False, hidden=hidden, lowrank=lowrank)
        module.hc_norm.weight = mx.zeros((width,), module.hc_norm.weight.dtype)
        up = module.input_mix_weight_up

        def attach_up(projection, k, n, dtype, inject, _up=up, _table=table):
            face = stock_face(projection, k, n, dtype, inject)
            return replace(face, tables=(_table,)) if projection is _up else face

        monkeypatch.setattr(live, "_projection_face", attach_up)
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
        live.rebuild_plan(layer)
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
