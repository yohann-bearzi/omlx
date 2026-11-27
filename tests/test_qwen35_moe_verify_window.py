# SPDX-License-Identifier: Apache-2.0
"""Row-exact verify windows through the fused routed-expert kernels.

Every row of a verify window must equal the fused one-token decode of that
row bit for bit (the served serial path), and must equal what the
verifier's composed MoE returned for the window before."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

import omlx.patches.qwen35_moe_routed_decode as routed
from omlx.formats import affine
from omlx.patches import qwen35_moe_router as router
from omlx.patches import qwen35_verify_qmm
from tests.test_m5_gather_qmm_nax import qwen_regroup as qwen_regroup
from tests.test_qwen35_moe_routed_decode import _plan
from tests.test_qwen35_moe_routed_decode import format_checkpoint as format_checkpoint

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

# Qwen3.8-Flash-Next oQ5e: 512 experts, top-10, hidden 2560, expert width 640.
HIDDEN, INTER, EXPERTS = 2560, 640, 512
# Quantized blocks keep the expert shape but not the count: 512 experts peak
# past the CI runner's memory. The fused router needs HIDDEN < 16 x experts
# and experts % 128 == 0.
BLOCK_EXPERTS = 256


class _FakeQwen4Model:
    pass


_FakeQwen4Model.__module__ = "mlx_vlm.models.qwen4_exp.qwen4_exp"


@pytest.fixture(autouse=True)
def _patched():
    """Served patch chain: verify linears, fused router, fused routed decode
    (which installs the verify-window entry). Restores the block class."""
    from mlx_vlm.models.qwen3_5_moe import language as vlm_moe

    from omlx.patches.mlx_vlm_mtp import qwen35_verify_linear

    qwen35_verify_qmm.apply_verify_qmm_patch()
    qwen35_verify_linear.apply()
    assert router.apply_qwen35_moe_router_patch()
    cls = vlm_moe.Qwen3_5MoeSparseMoeBlock
    call = cls.__call__
    had_flag = "_omlx_routed_decode" in cls.__dict__
    assert routed.apply_qwen35_moe_routed_decode_patch()
    yield
    qwen35_verify_qmm.set_verify_qmm_armed(False)
    cls.__call__ = call
    if not had_flag and "_omlx_routed_decode" in cls.__dict__:
        delattr(cls, "_omlx_routed_decode")


_BLOCKS: dict = {}


def _quantized_experts(experts, out_dims, in_dims, bits, dtype, chunk=32):
    """Stacked gs64 expert weights, quantized a few experts at a time so the
    FP32 draws never exceed one chunk."""
    scale = in_dims**-0.5
    parts = []
    for _ in range(0, experts, chunk):
        w = mx.random.uniform(-scale, scale, (chunk, out_dims, in_dims)).astype(dtype)
        parts.append(mx.quantize(w, 64, bits))
        mx.eval(parts[-1])
    return tuple(mx.concatenate(p) for p in zip(*parts, strict=True))


def _block(
    seed=0,
    bits=5,
    experts=BLOCK_EXPERTS,
    hidden=HIDDEN,
    inter=INTER,
    top_k=10,
    dtype=mx.bfloat16,
    shared_bits=8,
    shared_group=128,
    router_bits=None,
):
    """A real-shape oQ block: quantized routed experts, ``shared_bits``
    shared expert, 8-bit gs64 shared-expert gate, ``dtype`` router (8-bit
    gs64 quantized with ``router_bits``)."""
    key = (
        seed,
        bits,
        experts,
        hidden,
        inter,
        top_k,
        dtype,
        shared_bits,
        shared_group,
        router_bits,
    )
    if key in _BLOCKS:
        # A test proves its own launches: the plans are built again, with
        # proofs from this test's registry.
        block = _BLOCKS[key]
        block.__dict__.pop("_omlx_routed_decode_plan", None)
        block.switch_mlp.__dict__.pop("_omlx_verify_gather_plan", None)
        return block
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    from omlx.patches.qwen35_moe_gate_up import apply_qwen35_moe_gate_up_fusion

    mx.random.seed(seed)
    args = SimpleNamespace(
        hidden_size=hidden,
        moe_intermediate_size=inter,
        shared_expert_intermediate_size=inter,
        num_experts=experts,
        num_experts_per_tok=top_k,
        norm_topk_prob=True,
    )
    block = Qwen3_5MoeSparseMoeBlock(args)
    block.set_dtype(dtype)
    sm = block.switch_mlp
    for name in ("gate_proj", "up_proj", "down_proj"):
        layer = getattr(sm, name).to_quantized(64, bits)
        layer.weight, layer.scales, layer.biases = _quantized_experts(
            experts, layer.output_dims, layer.input_dims, bits, dtype
        )
        setattr(sm, name, layer)
    shared = block.shared_expert
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(
            shared,
            name,
            nn.QuantizedLinear.from_linear(
                getattr(shared, name), shared_group, shared_bits
            ),
        )
    block.shared_expert_gate = nn.QuantizedLinear.from_linear(
        block.shared_expert_gate, 64, 8
    )
    if router_bits:
        block.gate = nn.QuantizedLinear.from_linear(block.gate, 64, router_bits)
    block.eval()
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("mlp.switch_mlp", sm)]
    assert apply_qwen35_moe_gate_up_fusion(model) == 1
    mx.eval(block.parameters())
    _BLOCKS.clear()  # one real-shape block resident at a time
    _BLOCKS[key] = block
    return block


def _same_bits(a, b):
    return (
        a.shape == b.shape
        and mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()
    )


def _does_not_compile(a, shape=None):
    """An array like ``a`` (of ``shape``) that fails when it is evaluated: MLX
    compiles a kernel then."""
    kernel = mx.fast.metal_kernel(
        name="does_not_compile",
        input_names=["a"],
        output_names=["y"],
        source="does not compile;",
    )
    return kernel(
        inputs=[a],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[shape or a.shape],
        output_dtypes=[a.dtype],
    )[0]


def _serial(block, x):
    """The served one-token decode of every row (the fused routed call)."""
    rows = x.reshape(-1, 1, 1, x.shape[-1])
    assert _plan(block, rows[0]).launches.fold
    return mx.concatenate([block(row) for row in rows]).reshape(x.shape)


def _verify(block, x, window, row_exact=True):
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

    fused = routed.routed_verify_window
    if not window:
        routed.routed_verify_window = lambda block, x: None
    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=row_exact)
    try:
        out = Qwen3_5BatchInvariantForward()._feed_forward(block, x)
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)
        routed.routed_verify_window = fused
    mx.eval(out)
    return out


def _check(block, x, engaged, row_exact=True):
    """Window rows == one-token rows: the router's logits are its one-row
    arithmetic per row (``row_exact_linear`` over a quantized or FormatLinear
    gate), and the window-off rows agree with them."""
    count = len(engaged)
    new = _verify(block, x, window=True, row_exact=row_exact)
    ref = _serial(block, x)
    mx.eval(ref)
    assert _same_bits(new, ref)
    assert engaged[count:] == [True]
    if row_exact and x.size // x.shape[-1] <= router._MAX_ROWS:
        assert _same_bits(new, _verify(block, x, window=False))


@pytest.fixture
def engaged(monkeypatch):
    calls = []
    window = routed.routed_verify_window

    def spy(block, x):
        y = window(block, x)
        calls.append(y is not None)
        return y

    monkeypatch.setattr(routed, "routed_verify_window", spy)
    return calls


def _inputs(rows, step, batch=1, hidden=HIDDEN, dtype=mx.bfloat16):
    shape = (batch, rows, hidden)
    if step == 0:  # independent rows, mostly distinct experts
        return (mx.random.normal(shape) * 1.5).astype(dtype)
    if step == 1:  # consecutive-token-like rows sharing many experts
        base = mx.random.normal((batch, 1, hidden))
        return (0.97 * base + 0.25 * mx.random.normal(shape)).astype(dtype)
    # identical rows: every expert serves every row
    row = mx.random.normal((batch, 1, hidden)).astype(dtype)
    return mx.broadcast_to(row, shape)


@pytest.fixture(params=[True, False], ids=["topk_fold", "topk_launch"])
def topk_fold(request, monkeypatch):
    """Routing folded into the gate+up launch, or the separate routing launch."""
    if not request.param:
        monkeypatch.setattr(routed, "_topk_folds", lambda plan, logits: False)
    return request.param


def test_window_rows_equal_one_token_decode(topk_fold, engaged):
    block = _block(0)
    plan = _plan(block, mx.zeros((1, 1, HIDDEN), mx.bfloat16))
    assert plan.launches.topk_kernel is not None
    # Row bounds: 3 folded rows, 8 composed-router rows, 16 window rows.
    # Past 3 rows both variants take the routing launch.
    window_rows = (1, 3, 8, 16) if topk_fold else (1, 3)
    for rows in window_rows:
        for step in range(3):
            _check(block, _inputs(rows, step), engaged)
    # Batched verify rows (requests x drafts) are rows too.
    _check(block, _inputs(3, 1, batch=2), engaged)
    # A window of one row is one token's call (its proof, not the window's).
    assert plan.window.ran == {
        (rows, BLOCK_EXPERTS, topk_fold and rows <= routed._TOPK_FOLD_MAX_ROWS)
        for rows in (*window_rows, 6)
        if rows > 1
    }
    # One token's launches are proven beside the windows'.
    assert plan.proof.ran == {(BLOCK_EXPERTS, topk_fold)}
    assert not plan.proof.failed and not plan.window.failed


@pytest.mark.parametrize("bits", [2, 3])
def test_2_and_3_bit_window_rows_equal_one_token_decode(bits, engaged):
    """MLX's 2- and 3-bit experts take the window launches as the other
    widths do, gate+up through qmv_fast and down through qmv."""
    block = _block(0, bits=bits)
    for rows in (1, 2, 3, 4, 8, 16):
        for step in range(2):
            _check(block, _inputs(rows, step), engaged)


@pytest.mark.parametrize("rows", [3, 8])
def test_batched_decode_rows_equal_one_token_decode(rows, engaged):
    """Several requests' one-token decode ([B, 1, hidden], not a verify
    window) runs the window launches: every row equals its own fused
    one-token call bit for bit."""
    block = _block(0)
    x = _inputs(rows, 0).reshape(rows, 1, HIDDEN)
    count = len(engaged)
    got = block(x)
    ref = _serial(block, x)
    assert engaged[count:] == [True]
    assert _same_bits(got, ref.reshape(got.shape))


def test_router_ties_and_high_experts(engaged):
    """Exactly tied and one-ulp-apart router probabilities must select the
    same experts in the same order as the one-token router; experts at the
    end of the stacked weights are read past the bound one-expert view."""
    block = _block(0)
    weight = block.gate.weight
    experts = BLOCK_EXPERTS
    # Exact ties: experts / 8 distinct router rows, each repeated 8 times.
    tied = mx.concatenate([weight[: experts // 8]] * 8)[mx.random.permutation(experts)]
    # Near ties: experts / 16 rows repeated 16 times, each copy one weight
    # element about one bf16 ulp away, so logits tie or differ in the last bit.
    column = (mx.arange(experts) * 7) % HIDDEN
    bump = (mx.arange(HIDDEN)[None, :] == column[:, None]).astype(mx.float32) * 2**-13
    near = mx.concatenate([weight[: experts // 16]] * 16) + bump
    # High experts: the last 14 router rows lean on a direction the inputs
    # carry, so every row routes to the last 14 experts.
    direction = mx.random.normal((HIDDEN,))
    last = (mx.arange(experts) >= experts - 14)[:, None]
    high = weight + mx.where(last, 0.02 * direction, 0.0)
    try:
        for gate, lean in ((tied, 0.0), (near, 0.0), (high, 1.0)):
            block.gate.weight = gate.astype(weight.dtype)
            mx.eval(block.gate.weight)
            # Folded and launched routing, both within the composed router's rows.
            for rows in (3, 8):
                for step in range(3):
                    x = (_inputs(rows, step) + lean * direction).astype(mx.bfloat16)
                    _check(block, x, engaged)
                    if lean:
                        weight = block.gate.weight
                        logits = router.router_gemv(weight)(
                            x.reshape(rows, HIDDEN), weight
                        )
                        inds, _ = router.softmax_topk_rows(logits, 10)
                        assert mx.min(inds).item() >= experts - 14
    finally:
        block.gate.weight = weight  # The block stays cached for later tests.


def test_a_window_launch_that_does_not_compile_falls_back_to_the_composed_verifier(
    monkeypatch, engaged
):
    block = _block(0)
    x = _inputs(8, 1)
    old = _verify(block, x, window=False)
    plan = _plan(block, x[:, :1])
    assert not plan.proof.ran and not plan.window.ran
    calls = []

    def broken(launches, h, indices, scores, *operands):
        calls.append(1)
        return _does_not_compile(h, (indices.shape[0], launches.hidden))

    monkeypatch.setattr(routed, "_window_down", broken)
    for _ in range(2):
        out = _verify(block, x, window=True)
        assert engaged[-1] is False and plan.window.failed
        assert _same_bits(out, old)
    assert len(calls) == 1 and not plan.window.ran and not plan.proof.ran
    # One token still takes the fused launches.
    assert _plan(block, x[:, :1]) is plan
    assert _same_bits(_serial(block, x), old)
    assert plan.proof.ran == {(BLOCK_EXPERTS, True)} and not plan.proof.failed


def test_a_failure_ahead_of_a_window_is_raised_where_its_rows_are_evaluated(engaged):
    """A window is proven on zeros and built over its rows: their failure is
    the caller's, raised where the caller evaluates the result."""
    block = _block(0)
    x = _inputs(4, 1)
    plan = _plan(block, x[:, :1])
    y = routed.routed_verify_window(block, _does_not_compile(x))
    with pytest.raises(RuntimeError, match="Unable to build metal library"):
        mx.eval(y)
    assert engaged == [True] and not plan.window.failed
    assert plan.window.ran == {(4, BLOCK_EXPERTS, False)}
    _check(block, x, engaged)


def test_an_unproven_window_evaluates_its_stand_in_and_a_proven_one_nothing(
    monkeypatch,
):
    """A window's one evaluation is the stand-in of launches whose Metal is
    not proven: not its rows, not a tensor, not the result."""
    block = _block(0)
    x = _inputs(4, 1)
    _plan(block, x[:, :1])  # evaluates the experts' views
    evaluate, evaluated = mx.eval, []

    def recorded(*arrays):
        evaluated.append(arrays)
        evaluate(*arrays)

    monkeypatch.setattr(mx, "eval", recorded)
    y = routed.routed_verify_window(block, x)
    *inputs, (proving,) = evaluated
    assert len(proving) == 2 and not any(a is y for a in proving)
    # Ahead of it, the stand-in's logits and zeros: none of the call's.
    assert len(inputs) == 1 and not any(a is x for arrays in inputs for a in arrays)
    del evaluated[:]
    assert routed.routed_verify_window(block, x) is not None
    assert routed.routed_verify_window(block, _inputs(4, 0)) is not None
    assert not evaluated


def test_a_failed_one_token_launch_keeps_windows_composed(monkeypatch, engaged):
    block = _block(0)
    x = _inputs(4, 1)
    old = _verify(block, x, window=False)
    plan = _plan(block, x[:, :1])
    monkeypatch.setattr(plan.proof, "failed", True)
    assert _plan(block, x[:, :1]) is None
    out = _verify(block, x, window=True)
    assert engaged[-1] is False
    assert _same_bits(out, old)


def test_top8_window_and_batched_rows_equal_one_token_decode(engaged):
    """Qwen3.5/3.6-35B-A3B oQ layout (hidden 2048, intermediate 512, top-8):
    verify windows and batched decode rows equal the fused one-token decode
    of each row."""
    hidden, dtype = 2048, mx.float16
    block = _block(4, bits=4, hidden=hidden, inter=512, top_k=8, dtype=dtype)
    for rows in (1, 3, 8):
        for step in range(3):
            _check(block, _inputs(rows, step, hidden=hidden, dtype=dtype), engaged)
    x = _inputs(3, 0, hidden=hidden, dtype=dtype).reshape(3, 1, hidden)
    count = len(engaged)
    got = block(x)
    assert engaged[count:] == [True]
    assert _same_bits(got, _serial(block, x).reshape(got.shape))


def test_gate_up_qmv_window_and_batched_rows_equal_one_token_decode(engaged):
    """Gate+up as MLX's ``qmv`` (hidden 1152 at 4 bits ends in a partial
    kernel block): verify windows and batched decode rows equal the fused
    one-token decode of each row."""
    hidden = 1152
    block = _block(5, bits=4, hidden=hidden, inter=384, top_k=8)
    for rows in (1, 3, 8, 16):
        for step in range(3):
            _check(block, _inputs(rows, step, hidden=hidden), engaged)
    x = _inputs(5, 0, hidden=hidden).reshape(5, 1, hidden)
    count = len(engaged)
    got = block(x)
    assert engaged[count:] == [True]
    assert _same_bits(got, _serial(block, x).reshape(got.shape))


@pytest.mark.parametrize("row_exact", [True, False], ids=["row_exact", "verify"])
def test_quantized_router_window_rows_equal_one_token_decode(row_exact, engaged):
    """Qwen3.6-35B-A3B 4-bit layout (4-bit gs64 experts and shared expert,
    8-bit gs64 router and shared-expert gate): row-exact and ordinary verify
    windows and batched decode rows equal the fused one-token decode of each
    row."""
    hidden = 2048
    block = _block(
        6,
        bits=4,
        hidden=hidden,
        inter=512,
        top_k=8,
        shared_bits=4,
        shared_group=64,
        router_bits=8,
    )
    for rows in (1, 3, 8):
        for step in range(3):
            x = _inputs(rows, step, hidden=hidden)
            _check(block, x, engaged, row_exact=row_exact)
    x = _inputs(3, 0, hidden=hidden).reshape(3, 1, hidden)
    count = len(engaged)
    got = block(x)
    assert engaged[count:] == [True]
    assert _same_bits(got, _serial(block, x).reshape(got.shape))


def _quantized(shape, group_size, bits):
    return mx.quantize(mx.random.normal(shape) * 0.05, group_size, bits)


@pytest.mark.parametrize("window_rps,token_rps", [(4, 4), (4, 2)])
@pytest.mark.parametrize("bits", [2, 3, 4, 5])
def test_fp32_window_kernels_equal_one_token_kernels(bits, window_rps, token_rps):
    """BF16 outputs hide one-ulp FP32 differences: run the window launches
    and the one-token launches in FP32 (the one-token ones equal MLX's FP32
    mat-vecs, see test_qwen35_moe_routed_decode) and compare row by row. The
    rows share experts at different slots and read the last experts. The
    down launches group rows per simdgroup as served (``_down_rows``): a
    verify window must equal one-token decode under either grouping."""
    f32, top_k, experts, rows = mx.float32, 10, 16, 4
    mx.random.seed(60 + bits)
    fmt = partial(affine.face, dtype=f32)
    routed_gu, shared_gu, gate_fmt = (
        fmt(bits, 64, True),
        fmt(8, 128, True),
        fmt(8, 64, False),
    )
    routed_d, shared_d = fmt(bits, 64, False), fmt(8, 128, False)
    experts_gate_up = _quantized((experts, 2 * INTER, HIDDEN), 64, bits)
    experts_down = _quantized((experts, HIDDEN, INTER), 64, bits)
    shared = _quantized((INTER, HIDDEN), 128, 8) + _quantized((INTER, HIDDEN), 128, 8)
    gate_row = _quantized((1, HIDDEN), 64, 8)
    shared_down = _quantized((HIDDEN, INTER), 128, 8)
    first = mx.random.permutation(experts)[:top_k]
    ids = mx.stack(
        [
            first,
            first[::-1],  # same experts, other slots
            mx.concatenate([first[:5], mx.arange(experts - 5, experts)]),
            mx.arange(experts - top_k, experts)[::-1],  # the last experts
        ]
    ).astype(mx.uint32)
    x = mx.random.normal((rows, HIDDEN))
    scores = mx.softmax(mx.random.normal((rows, top_k)), axis=-1)
    gate_up_template = [
        ("T", f32),
        ("K", HIDDEN),
        ("NI", INTER),
        ("RPS", 2),
        ("NSG", 2),
        ("NS", INTER),
        ("TOPK", top_k),
    ]
    blocks = 1 + INTER // 4 + top_k * INTER // 4
    width = top_k * INTER + INTER + 1
    h = routed._gate_up_window_kernel(routed_gu, shared_gu, gate_fmt)(
        inputs=[x, *experts_gate_up, ids, *shared, *gate_row],
        template=gate_up_template + [("M", rows)],
        grid=(32, 2 * blocks * rows, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(rows, width)],
        output_dtypes=[f32],
    )[0]
    down_template = [
        ("T", f32),
        ("K", INTER),
        ("N", HIDDEN),
        ("KS", INTER),
        ("NPART", top_k + 1),
        ("TOPK", top_k),
    ]
    y = routed._down_window_kernel(routed_d, shared_d)(
        inputs=[h, *experts_down, *shared_down, ids, scores],
        template=down_template + [("RPS", window_rps), ("M", rows)],
        grid=(32, (top_k + 1) * HIDDEN // window_rps * rows, 1),
        threadgroup=(32, top_k + 1, 1),
        output_shapes=[(rows, HIDDEN)],
        output_dtypes=[f32],
    )[0]
    for r in range(rows):
        h_r = routed._gate_up_kernel(routed_gu, shared_gu, gate_fmt)(
            inputs=[x[r], *experts_gate_up, ids[r], *shared, *gate_row],
            template=gate_up_template,
            grid=(32, 2 * blocks, 1),
            threadgroup=(32, 2, 1),
            output_shapes=[(width,)],
            output_dtypes=[f32],
        )[0]
        y_r = routed._down_kernel(routed_d, shared_d)(
            inputs=[h_r, *experts_down, *shared_down, ids[r], scores[r]],
            template=down_template + [("RPS", token_rps)],
            grid=(32, (top_k + 1) * HIDDEN // token_rps, 1),
            threadgroup=(32, top_k + 1, 1),
            output_shapes=[(HIDDEN,)],
            output_dtypes=[f32],
        )[0]
        assert mx.array_equal(h[r].view(mx.uint32), h_r.view(mx.uint32)).item()
        assert mx.array_equal(y[r].view(mx.uint32), y_r.view(mx.uint32)).item()


def test_fp32_window_router_equals_one_row_router():
    """The router gemv and softmax + top-k of a window, in FP32, row by row
    against the one-row launches (which reproduce MLX's)."""
    f32, rows = mx.float32, 5
    mx.random.seed(71)
    weight = mx.random.normal((EXPERTS, HIDDEN)) * 0.02
    # A few duplicated router rows give exactly tied probabilities.
    weight = mx.concatenate([weight[:480], weight[:32]])
    x = mx.random.normal((rows, HIDDEN))
    halves = weight.astype(mx.bfloat16)
    mx.eval(router.router_gemv(halves)(x.astype(mx.bfloat16), halves))

    def gemv(x, rows):
        return router._GEMV_KERNEL(
            inputs=[x, weight],
            template=[
                ("T", f32),
                ("K", HIDDEN),
                ("N", EXPERTS),
                ("M", rows),
                ("NSG", 4),
            ],
            grid=(32, EXPERTS * rows, 1),
            threadgroup=(32, 4, 1),
            output_shapes=[(rows, EXPERTS)],
            output_dtypes=[f32],
        )[0]

    logits = gemv(x, rows)
    probe = mx.zeros((rows, EXPERTS), mx.bfloat16)
    router.softmax_topk_row(probe[:1], 10)
    router.softmax_topk_rows(probe, 10)
    inds, scores = router._SOFTMAX_TOPK_ROWS_KERNEL(
        inputs=[logits],
        template=[("T", f32), ("NE", EXPERTS), ("K", 10)],
        grid=(32, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, 10), (rows, 10)],
        output_dtypes=[mx.uint32, f32],
    )
    for r in range(rows):
        logits_r = gemv(x[r], 1)[0]
        inds_r, scores_r = router._SOFTMAX_TOPK_KERNEL(
            inputs=[logits_r],
            template=[("T", f32), ("NE", EXPERTS), ("K", 10)],
            grid=(32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(10,), (10,)],
            output_dtypes=[mx.uint32, f32],
        )
        assert mx.array_equal(
            logits[r].view(mx.uint32), logits_r.view(mx.uint32)
        ).item()
        assert mx.array_equal(inds[r], inds_r).item()
        assert mx.array_equal(
            scores[r].view(mx.uint32), scores_r.view(mx.uint32)
        ).item()


def test_a_face_with_a_table_runs_the_window_launches_to_the_bit(monkeypatch, engaged):
    """The routed experts' 4-bit face reads its weight values from a table
    that the window's launches stage: their rows are the one-token rows and
    the composed verifier's."""
    from omlx.formats import kernels
    from tests.test_formats_tables import tabled_dot

    tabled_dot(monkeypatch)
    block = _block(0, bits=4)
    for rows in (2, 4):
        _check(block, _inputs(rows, 1), engaged)
    plan = _plan(block, _inputs(1, 0))
    assert plan.window.ran and not plan.window.failed
    assert any(
        key[1].key.endswith("_lut")
        for key in kernels._PROOFS
        if key[0] == "routed_window"
    )


@pytest.mark.usefixtures("format_checkpoint")
def test_a_window_of_format_layers_equals_one_token_decode(monkeypatch, engaged):
    """A block of quant format layers takes the window launches as MLX's
    quantized layers do."""
    from tests.test_formats_kernels import _toy
    from tests.test_qwen35_moe_routed_decode import format_block, load_toy_format

    load_toy_format(monkeypatch, replace(_toy("window"), dtypes=(mx.bfloat16,)))
    # The fused router needs hidden < 16 x experts and experts % 128 == 0.
    block = format_block(1024, 320, "toy", experts=128)
    x = _inputs(4, 0, hidden=1024)
    assert routed.routed_verify_window(block, x) is not None
    for rows in (1, 2, 3, 8):
        for step in range(2):
            _check(block, _inputs(rows, step, hidden=1024), engaged)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("rows", [2, 3, 4])
def test_coalesced_window_kernels_equal_one_token_decode(
    dtype, rows, monkeypatch, tmp_path
):
    """Pair-coalesced verify-window kernels (3-launch folded and 4-launch
    meta-driven, decoding each shared expert block once for two rows via the
    dot face's decode/apply contract) and the unrolled window down kernel
    reproduce one-token decode bit for bit across dtypes and overlap regimes."""
    # The coalesced path ships for the measured shape only (Flash-Next's
    # windows); the other activation type takes the same shape, and a shape
    # outside the allowlist is covered by its own gating test.
    from omlx.formats import tuning
    from tests.test_formats_tables import tabled_dot

    tabled_dot(monkeypatch)
    chip = "applegpu_g17s_c40_bothcoal"
    v1_dir = tmp_path / "v1"
    monkeypatch.setattr(tuning, "_TABLE_DIRS", [v1_dir])
    monkeypatch.setattr(tuning, "_EXTRA_DIRS", [])
    monkeypatch.setattr(tuning, "current_chip", lambda: chip)
    _write_routed_window_table(
        v1_dir, chip, {2: "folded", 3: "unfolded", 4: "unfolded"}
    )
    tuning.reset_cache()
    try:
        hidden, inter, top_k = 2560, 640, 10
        block = _block(7, bits=4, hidden=hidden, inter=inter, top_k=top_k, dtype=dtype)
        plan = _plan(block, mx.zeros((1, 1, hidden), dtype))
        launches = plan.launches
        assert launches.window_fast
        assert launches.window_coalesced_kernel is not None
        assert launches.window_topk_coalesced_kernel is not None
        assert launches.window_down_unrolled_kernel is not None
        w, gu_op, dn_op, sgu_op, sdn_op = plan.operands
        for step in range(3):
            x = _inputs(rows, step, hidden=hidden, dtype=dtype)
            ref = _serial(block, x).reshape(rows, hidden)
            x2d = mx.reshape(x, (rows, hidden))
            logits = launches.router(x2d, *w)
            # 4-launch coalesced gate+up + unrolled down
            ind, sc, meta, ucount = router.softmax_topk_coalesce(logits, top_k)
            h_c2 = routed._window_gate_up_coalesced(
                launches, x2d, meta, ucount, gu_op, sgu_op
            )
            y_c2 = routed._window_down(launches, h_c2, ind, sc, dn_op, sdn_op)
            # 3-launch folded pair-coalesced gate+up + unrolled down
            h_fc, ind_fc, sc_fc = routed._window_gate_up_topk_coalesced(
                launches, x2d, logits, gu_op, sgu_op
            )
            y_fc = routed._window_down(launches, h_fc, ind_fc, sc_fc, dn_op, sdn_op)
            mx.eval(ref, y_c2, y_fc, ind, ind_fc, sc, sc_fc)
            assert mx.array_equal(ind, ind_fc).item()
            assert _same_bits(sc, sc_fc)
            assert _same_bits(y_c2, ref)
            assert _same_bits(y_fc, ref)
    finally:
        tuning.reset_cache()


def test_coalesced_window_kernels_engage_for_tabled_face_and_have_no_dead_code(
    monkeypatch, engaged, tmp_path
):
    """A tabled dot face engages the coalesced window kernels for configured
    single-stream rows (matching one-token serial decode bit for bit), and the
    new builders carry no uncalled Metal functions or hand-written lookup
    tables."""
    from omlx.formats import tuning
    from tests.test_formats_tables import tabled_dot
    from tests.test_metal_tables import hand_tables
    from tests.test_qwen35_moe_routed_decode import uncalled

    built = {}
    make = mx.fast.metal_kernel

    def spy_kernel(**spec):
        built[spec["name"]] = (spec.get("header", ""), spec["source"])
        return make(**spec)

    monkeypatch.setattr(mx.fast, "metal_kernel", spy_kernel)
    for builder in (
        routed._gate_up_window_coalesced_kernel,
        routed._gate_up_topk_coalesced_kernel,
        routed._down_window_unrolled_kernel,
        router._softmax_topk_coalesce_kernel,
    ):
        builder.cache_clear()
    tabled_dot(monkeypatch)
    chip = "applegpu_g15d_c80_deadcode"
    v1_dir = tmp_path / "v1"
    monkeypatch.setattr(tuning, "_TABLE_DIRS", [v1_dir])
    monkeypatch.setattr(tuning, "_EXTRA_DIRS", [])
    monkeypatch.setattr(tuning, "current_chip", lambda: chip)
    _write_routed_window_table(v1_dir, chip, {2: "folded", 3: "none", 4: "unfolded"})
    tuning.reset_cache()
    try:
        block = _block(0, bits=4)
        for rows in (2, 3, 4):
            for step in range(3):
                _check(block, _inputs(rows, step), engaged)
    finally:
        tuning.reset_cache()
    names = list(built)
    assert any("gate_up_window_coalesced" in n for n in names), names
    assert any("gate_up_topk_coalesced" in n for n in names), names
    assert any("down_combine_window_unrolled" in n for n in names), names
    for name, (header, source) in built.items():
        if name.startswith("omlx_qwen35_moe_gate_up_") or name.startswith(
            "omlx_qwen35_moe_down_"
        ):
            assert uncalled(header + source) == [], name
    # Table-free affine builds of the new builders carry no hand-written tables.
    face = partial(affine.face, dtype=mx.bfloat16)
    gu, dn, sh, gt = (
        face(4, 64, True),
        face(4, 64, False),
        face(8, 128, True),
        face(8, 64, True),
    )
    built.clear()
    routed._gate_up_window_coalesced_kernel.__wrapped__(gu, sh, gt)
    routed._gate_up_topk_coalesced_kernel.__wrapped__(gu, sh, gt, mx.bfloat16)
    routed._down_window_unrolled_kernel.__wrapped__(dn, sh)
    for name, (header, source) in built.items():
        assert uncalled(header + source) == [], name
        assert not hand_tables(header), name
        assert not hand_tables(source), name


def test_mixed_dtype_quantized_router_skips_window_plan(monkeypatch):
    """A quantized gate whose scales are not the activation dtype is not a
    window router: ``row_exact_linear`` would promote in its tile and every
    window would decline after paying the gate's work."""
    hidden = 2048
    block = _block(
        8,
        bits=4,
        hidden=hidden,
        inter=512,
        top_k=8,
        shared_bits=4,
        shared_group=64,
        router_bits=8,
    )
    block.gate.scales = block.gate.scales.astype(mx.float16)
    block.gate.biases = block.gate.biases.astype(mx.float16)
    mx.eval(block.parameters())
    plan = _plan(block, _inputs(1, 0, hidden=hidden))
    assert plan is not None and plan.quantized_router is False
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("row_exact_linear ran for a mixed-dtype gate")

    monkeypatch.setattr(routed, "row_exact_linear", forbidden)
    assert routed.routed_verify_window(block, _inputs(2, 0, hidden=hidden)) is None
    assert calls == []


def test_a_shared_router_less_proof_keeps_the_window_variants_apart():
    """A quantized-router plan and a declined plain gate can share the window
    proof (the router's absence is in its key). The kept graph still names its
    variant: the declined plan must not replay the external-logits graph with
    two arguments."""
    hidden = 2048
    quantized = _block(
        6,
        bits=4,
        hidden=hidden,
        inter=512,
        top_k=8,
        shared_bits=4,
        shared_group=64,
        router_bits=8,
    )
    plain = _block(
        7, bits=4, hidden=hidden, inter=512, top_k=8, shared_bits=4, shared_group=64
    )
    plain.gate.bias = mx.zeros((plain.num_experts,), mx.bfloat16)
    mx.eval(plain.parameters())
    rows = _inputs(3, 0, hidden=hidden)
    assert routed.routed_verify_window(quantized, rows) is not None
    q_plan = _plan(quantized, _inputs(1, 0, hidden=hidden))
    p_plan = _plan(plain, _inputs(1, 0, hidden=hidden))
    assert p_plan is not None and p_plan.quantized_router is False
    assert q_plan.window is p_plan.window
    assert routed.routed_verify_window(plain, rows) is None


def test_softmax_topk_coalesce_rejects_bad_bounds():
    """The coalesced routing launch declines bounds its threadgroup cannot
    hold or its selection cannot index: top_k < 1 and more than 32 rows."""
    logits = mx.zeros((2, 128), mx.bfloat16)
    assert router.softmax_topk_coalesce(logits, 0) is None
    assert router.softmax_topk_coalesce(logits, -1) is None
    assert (
        router.softmax_topk_coalesce(mx.zeros((33, 128), mx.bfloat16), 1, max_rows=64)
        is None
    )
    # The accepted boundaries stay accepted.
    assert (
        router.softmax_topk_coalesce(mx.zeros((32, 128), mx.bfloat16), 1, max_rows=32)
        is not None
    )
    assert router.softmax_topk_coalesce(logits, 1) is not None


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_the_router_coalesce_kernel_carries_only_its_own_selection(dtype, monkeypatch):
    """The coalesced routing launch compiles one dtype's selection core: no
    ``if constexpr`` arm of the other dtype, no ``nth``/``write`` and no store
    helper (a C++17 caller would have to resolve its name), and nothing its
    source does not call."""
    from tests.test_formats_affine import uncalled

    built = {}
    make = mx.fast.metal_kernel

    def spy_kernel(**spec):
        built[spec["name"]] = (spec.get("header", ""), spec["source"])
        return make(**spec)

    monkeypatch.setattr(mx.fast, "metal_kernel", spy_kernel)
    router._softmax_topk_coalesce_kernel.cache_clear()
    assert (
        router.softmax_topk_coalesce(mx.zeros((2, 128), dtype), 8, max_rows=8)
        is not None
    )
    name, (header, source) = next(iter(built.items()))
    assert "softmax_topk_coalesce" in name
    assert "if constexpr (metal::is_same_v<T, bfloat16_t>)" not in header
    assert "omlx_router_topk_store" not in header
    assert "void write(" not in header
    assert "uint nth(" not in header
    assert (
        "if constexpr (metal::is_same_v<T, bfloat16_t>)"
        not in router.topk_indices_header(dtype)
    )
    assert uncalled(header + source) == [], name


def _stock_affine_format_layer(linear):
    import json
    import pathlib
    import tempfile

    from omlx.formats import testing as fmt_testing
    from omlx.formats.linear import FormatLinear
    from omlx.patches.format_load import loading

    directory = pathlib.Path(tempfile.mkdtemp())
    (directory / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "omlx_quant_formats": {"affine": {}}})
    )
    arrays = {
        "weight": linear["weight"],
        "scales": linear["scales"],
        "biases": linear["biases"],
    }
    with fmt_testing.verified("qwen4_exp"), loading(directory):
        return FormatLinear(
            "affine",
            linear.bits,
            linear.group_size,
            arrays["scales"].shape[1] * linear.group_size,
            arrays,
        )


@pytest.mark.parametrize("row_exact", [True, False], ids=["row_exact", "verify"])
def test_stock_affine_format_router_takes_the_fused_window(row_exact, engaged):
    """A stock checkpoint's router gate is a built-in affine FormatLinear;
    the verify window must take it through the fused gate instead of one
    stock call a row."""
    hidden = 2048
    block = _block(
        6,
        bits=4,
        hidden=hidden,
        inter=512,
        top_k=8,
        shared_bits=4,
        shared_group=64,
        router_bits=8,
    )
    block.gate = _stock_affine_format_layer(block.gate)
    block.eval()
    mx.eval(block.parameters())
    x = _inputs(5, 0, hidden=hidden).reshape(5, 1, hidden)
    _check(block, x, engaged, row_exact=row_exact)


@pytest.mark.parametrize("row_exact", [True, False], ids=["row_exact", "verify"])
def test_format_router_window_row_exact_under_msg_tile(row_exact, engaged, monkeypatch):
    """A FormatLinear MoE router feeds discrete top-k selection, so verify
    windows must run its one-row arithmetic per row (``row_exact_linear``)
    even when the dense schedule picks ``msg_tile`` for 4 rows (HIGH-1)."""
    from omlx.formats import linear, tile_dense, tuning
    from omlx.formats.tuning import MsgTilePolicy
    from omlx.patches.row_exact_qmv import quantized_linear as row_exact_linear

    hidden = 2560
    block = _block(
        7,
        bits=4,
        hidden=hidden,
        inter=512,
        top_k=8,
        shared_bits=4,
        shared_group=64,
        router_bits=8,
    )
    # Mirror both halves of the 8-bit router weights and feed cancelling
    # [half, -half] rows: every expert's true logit sits near 0 while its
    # half-sums are large, so msg_tile's different FP32 accumulation order
    # flips the discrete top-k expert set relative to 1-row Dot.
    w_half = (
        mx.random.normal((block.num_experts, hidden // 2), key=mx.random.key(71))
        * 512.0
    ).astype(mx.bfloat16)
    w_full = mx.concatenate([w_half, w_half], axis=-1)
    q_w, q_s, q_b = mx.quantize(w_full, group_size=64, bits=8)
    block.gate["weight"], block.gate["scales"], block.gate["biases"] = q_w, q_s, q_b
    block.gate = _stock_affine_format_layer(block.gate)
    block.eval()
    mx.eval(block.parameters())

    def lookup(route, *args, **kwargs):
        if route == "msg_tile" and kwargs.get("rows") == 4:
            return MsgTilePolicy(nsg=8)
        return None

    monkeypatch.setattr(tuning, "lookup", lookup)
    half = mx.random.normal((1, 4, hidden // 2), key=mx.random.key(77)).astype(
        mx.bfloat16
    )
    x = mx.concatenate([half, -half], axis=-1)
    # Control: calling gate directly at 4 rows takes msg_tile and differs in
    # bits from one-row Dot arithmetic (proving the window cannot bypass
    # row_exact_linear).
    flat = x.reshape(4, hidden)
    direct = block.gate(flat)
    exact = row_exact_linear(block.gate, flat)
    mx.eval(direct, exact)
    kept = vars(block.gate)[linear._KEPT]
    assert tile_dense.route_for(kept.dense, 4, x.dtype).backend == "msg"
    assert not _same_bits(direct, exact), "control: msg_tile must differ from 1-row Dot"
    # Cold build and warm graph replay must both keep exact one-row router logits.
    _check(block, x, engaged, row_exact=row_exact)
    _check(block, x, engaged, row_exact=row_exact)


def test_biased_format_router_declines_and_a_plugin_one_is_accepted():
    """Rule 25/D11: a plain bias-free FormatLinear router is taken whatever
    its format; a bias makes it decline."""
    from omlx.patches.qwen35_moe_routed_decode import _window_router_ok

    block = _block(
        6,
        bits=4,
        hidden=2048,
        inter=512,
        top_k=8,
        shared_bits=4,
        shared_group=64,
        router_bits=8,
    )
    gate = _stock_affine_format_layer(block.gate)
    assert _window_router_ok(gate, mx.bfloat16)

    gate["bias"] = mx.zeros((6,), dtype=mx.bfloat16)
    assert not _window_router_ok(gate, mx.bfloat16)

    plugin = _stock_affine_format_layer(block.gate)
    object.__setattr__(plugin, "fmt", object())
    assert _window_router_ok(plugin, mx.bfloat16)


def test_vlm_verify_router_respects_format_linear_subclass(monkeypatch):
    """pre-kit-82917ced LOW 3: a FormatLinear subclass overriding __call__
    must route through self._linear (preserving its override), never bypass
    into row_exact_linear."""
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

    from omlx.formats.linear import FormatLinear

    sub_calls = []

    class _OffsetFormatLinear(FormatLinear):
        def __call__(self, x):
            sub_calls.append(x.shape)
            return super().__call__(x) + mx.array(1.0, dtype=x.dtype)

    hidden = 256
    block = _block(
        8,
        bits=4,
        hidden=hidden,
        inter=128,
        top_k=4,
        shared_bits=4,
        shared_group=64,
        router_bits=8,
    )
    base_gate = _stock_affine_format_layer(block.gate)
    sub_gate = _OffsetFormatLinear.__new__(_OffsetFormatLinear)
    sub_gate._init_tensors(
        base_gate.fmt,
        base_gate.bits,
        base_gate.group_size,
        base_gate.input_dims,
        {name: base_gate[name] for name in base_gate._tensors},
        None,
        layout=base_gate.fmt.layout(
            base_gate.input_dims,
            base_gate.output_dims,
            base_gate.bits,
            base_gate.group_size,
        ),
    )
    block.gate = sub_gate
    block.eval()
    mx.eval(block.parameters())

    router._ensure_vlm_verify_patch()
    x = _inputs(2, 0, hidden=hidden)

    linear_calls = []
    orig_linear = Qwen3_5BatchInvariantForward._linear

    def spy_linear(self, layer, inp):
        linear_calls.append(layer)
        return orig_linear(self, layer, inp)

    monkeypatch.setattr(Qwen3_5BatchInvariantForward, "_linear", spy_linear)
    monkeypatch.setattr(routed, "routed_verify_window", lambda b, inp: None)

    out = Qwen3_5BatchInvariantForward()._feed_forward(block, x)
    mx.eval(out)
    assert any(layer is sub_gate for layer in linear_calls)
    assert len(sub_calls) >= 1

    # Negative control: plain FormatLinear still routes through row_exact_linear
    # (never via self._linear).
    linear_calls.clear()
    block.gate = base_gate
    out_plain = Qwen3_5BatchInvariantForward()._feed_forward(block, x)
    mx.eval(out_plain)
    assert not any(layer is base_gate for layer in linear_calls)


def _write_routed_window_table(v1_dir, chip, modes_by_row, *, bits=4, group=64):
    import json

    from omlx.formats import tuning

    if not hasattr(routed, "_ensure_tuning_registered"):
        return
    routed._ensure_tuning_registered()
    skel = tuning.skeleton_digest("routed_window")
    entries = []
    for rows, mode in modes_by_row.items():
        for dtype_name in ("bf16", "fp16"):
            entries.append(
                {
                    "key": {
                        "route": "routed_window",
                        "face_id": "*:lut_small",
                        "dtype": dtype_name,
                        "bits": bits,
                        "group": group,
                        "rows_bucket": str(rows),
                        "n": 2 * INTER,
                        "k": HIDDEN,
                    },
                    "face_sha256": "",
                    "skeleton_sha256": skel,
                    "measured_at": "2026-11-24T20:00:00Z",
                    "exactness": "bitwise",
                    "params": {"coalesce": mode},
                }
            )
    v1_dir.mkdir(parents=True, exist_ok=True)
    (v1_dir / f"{chip}.json").write_text(
        json.dumps({"schema_version": 1, "chip": chip, "entries": entries})
    )


def test_batched_decode_t1_never_coalesces(monkeypatch, tmp_path):
    """Independent batched decode (B > 1, T == 1) and multi-stream verify
    (B > 1, T > 1) must NEVER enter pair-coalescing; coalescing applies only
    to single-stream verify windows (B == 1, T > 1) when enabled by per-chip
    tuning data."""
    from omlx.formats import tuning
    from tests.test_formats_tables import tabled_dot

    tabled_dot(monkeypatch)
    chip = "applegpu_g17s_c40_t1test"
    v1_dir = tmp_path / "v1"
    monkeypatch.setattr(tuning, "_TABLE_DIRS", [v1_dir])
    monkeypatch.setattr(tuning, "_EXTRA_DIRS", [])
    monkeypatch.setattr(tuning, "current_chip", lambda: chip)
    _write_routed_window_table(
        v1_dir, chip, {2: "folded", 3: "unfolded", 4: "unfolded"}
    )
    tuning.reset_cache()
    try:
        block = _block(7, bits=4)
        events: list[tuple[str, int]] = []

        def wrap(mod, name):
            orig = getattr(mod, name)

            def spy(*args, **kwargs):
                rows = (
                    args[1].shape[0]
                    if name != "softmax_topk_coalesce"
                    else args[0].shape[0]
                )
                events.append((name, int(rows)))
                return orig(*args, **kwargs)

            monkeypatch.setattr(mod, name, spy)

        wrap(routed, "_window_gate_up_coalesced")
        wrap(routed, "_window_gate_up_topk_coalesced")
        wrap(routed, "softmax_topk_coalesce")
        wrap(routed, "_window_gate_up")
        wrap(routed, "_gate_up_topk")

        # 1. Batched one-token decode (B > 1, T == 1) via patched_call AND direct
        #    routed_verify_window must NEVER coalesce.
        for batch in (2, 3, 4):
            x = _inputs(batch, batch).reshape(batch, 1, HIDDEN)
            ref = _serial(block, x)
            events.clear()
            got_call = block(x)
            mx.eval(got_call, ref)
            assert _same_bits(got_call, ref)
            names = [n for n, _ in events]
            assert "_window_gate_up_coalesced" not in names, (batch, events)
            assert "_window_gate_up_topk_coalesced" not in names, (batch, events)
            assert "softmax_topk_coalesce" not in names, (batch, events)
            expected = "_gate_up_topk" if batch <= 3 else "_window_gate_up"
            assert expected in names, (batch, events)

            # Direct routed_verify_window call on (B, 1, hidden) replays the
            # uncoalesced graph and must not coalesce either.
            block.__dict__.pop("_omlx_routed_decode_plan", None)
            events.clear()
            got_win = routed.routed_verify_window(block, x)
            assert got_win is not None
            mx.eval(got_win)
            assert _same_bits(got_win, ref.reshape(batch, HIDDEN))
            names = [n for n, _ in events]
            assert "_window_gate_up_coalesced" not in names, (batch, events)
            assert "_window_gate_up_topk_coalesced" not in names, (batch, events)
            assert "softmax_topk_coalesce" not in names, (batch, events)

        # 2. Multi-stream verify (B == 2, T == 2) must NOT coalesce either.
        x_multi = _inputs(4, 11).reshape(2, 2, HIDDEN)
        ref_multi = _serial(block, x_multi).reshape(4, HIDDEN)
        events.clear()
        got_multi = routed.routed_verify_window(block, x_multi)
        assert got_multi is not None
        mx.eval(got_multi, ref_multi)
        assert _same_bits(got_multi, ref_multi)
        names = [n for n, _ in events]
        assert "_window_gate_up_coalesced" not in names, events
        assert "_window_gate_up_topk_coalesced" not in names, events
        assert "softmax_topk_coalesce" not in names, events
        assert "_window_gate_up" in names, events

        # 3. Single-stream verify (B == 1, T > 1) DOES coalesce per tuning data.
        for t, expect in (
            (2, "_window_gate_up_topk_coalesced"),
            (3, "_window_gate_up_coalesced"),
            (4, "_window_gate_up_coalesced"),
        ):
            x_ss = _inputs(t, 20 + t)
            assert x_ss.shape == (1, t, HIDDEN)
            ref_ss = _serial(block, x_ss).reshape(t, HIDDEN)
            events.clear()
            got_ss = routed.routed_verify_window(block, x_ss)
            assert got_ss is not None
            mx.eval(got_ss, ref_ss)
            assert _same_bits(got_ss, ref_ss)
            names = [n for n, _ in events]
            assert names.count(expect) == 2, (t, events)

        # 4. Quantized-router blocks (_window_logits) obey the same rule.
        qblock = _block(8, bits=4, router_bits=8)
        xq_batch = _inputs(4, 30).reshape(4, 1, HIDDEN)
        events.clear()
        got_qb = routed.routed_verify_window(qblock, xq_batch)
        assert got_qb is not None
        mx.eval(got_qb)
        assert "_window_gate_up_coalesced" not in [n for n, _ in events], events
        assert "_window_gate_up" in [n for n, _ in events], events

        xq_ss = _inputs(4, 31)
        events.clear()
        got_qs = routed.routed_verify_window(qblock, xq_ss)
        assert got_qs is not None
        mx.eval(got_qs)
        assert [n for n, _ in events].count("_window_gate_up_coalesced") == 2, events
    finally:
        tuning.reset_cache()


def test_routed_window_coalescing_uses_per_chip_data_not_gpu_generation(
    monkeypatch, tmp_path
):
    """Per-chip coalescing decisions come from omlx.formats.tuning DATA (rule 27),
    never a hand gpu_generation() predicate, and unused coalesced kernels are
    never constructed."""
    from omlx.formats import kernels, tuning
    from tests.test_formats_tables import tabled_dot

    tabled_dot(monkeypatch)
    v1_dir = tmp_path / "v1"
    chip_fast = "applegpu_g17s_c40_test"
    chip_off = "applegpu_g15d_c80_test"
    chip_unknown = "applegpu_g99x_c1_unknown"
    monkeypatch.setattr(tuning, "_TABLE_DIRS", [v1_dir])
    monkeypatch.setattr(tuning, "_EXTRA_DIRS", [])
    _write_routed_window_table(v1_dir, chip_fast, {2: "none", 3: "none", 4: "unfolded"})
    _write_routed_window_table(v1_dir, chip_off, {2: "none", 3: "none", 4: "none"})
    monkeypatch.setattr(
        kernels,
        "gpu_generation",
        lambda: (_ for _ in ()).throw(AssertionError("hand predicate forbidden")),
    )
    active = [chip_fast]
    monkeypatch.setattr(tuning, "current_chip", lambda: active[0])
    tuning.reset_cache()
    try:
        block = _block(7, bits=4)
        x1 = _inputs(1, 0)
        plan_fast = _plan(block, x1)
        assert plan_fast is not None
        assert plan_fast.launches.window_fast
        assert plan_fast.launches.window_coalesced
        assert plan_fast.launches.window_coalesced_kernel is not None
        assert plan_fast.launches.window_topk_coalesced_kernel is None

        # Run single-stream M=4 (coalesces) and batched (4, 1) (no coalesce),
        # with gpu_generation() poisoned.
        assert routed.routed_verify_window(block, _inputs(4, 1)) is not None
        assert (
            routed.routed_verify_window(block, _inputs(4, 2).reshape(4, 1, HIDDEN))
            is not None
        )

        # Switch to M3-like table where coalescing is disabled: no coalesced
        # kernels built, distinct window proof.
        active[0] = chip_off
        tuning.reset_cache()
        block.__dict__.pop("_omlx_routed_decode_plan", None)
        plan_off = _plan(block, x1)
        assert plan_off is not None
        assert plan_off.launches.window_fast
        assert not plan_off.launches.window_coalesced
        assert plan_off.launches.window_coalesced_kernel is None
        assert plan_off.launches.window_topk_coalesced_kernel is None
        assert plan_off.window is not plan_fast.window
        assert routed.routed_verify_window(block, _inputs(4, 3)) is not None

        # Unknown chip takes the parity floor (no coalescing).
        active[0] = chip_unknown
        tuning.reset_cache()
        block.__dict__.pop("_omlx_routed_decode_plan", None)
        plan_unk = _plan(block, x1)
        assert plan_unk is not None
        assert not plan_unk.launches.window_coalesced
        assert plan_unk.launches.window_coalesced_kernel is None
        assert plan_unk.launches.window_topk_coalesced_kernel is None
    finally:
        tuning.reset_cache()


def test_coalescing_disabled_when_experts_not_softmax_topk_eligible(
    monkeypatch, tmp_path
):
    """pre-p3-d32d3543 Finding 3: experts not divisible by 128 (or <128 / >4096)
    cannot use softmax_topk_coalesce, so coalescing must stay disabled and
    unused coalesced kernels must not be constructed or warmed."""
    from omlx.formats import tuning
    from tests.test_formats_tables import tabled_dot

    tabled_dot(monkeypatch)
    chip = "applegpu_g17s_c40_e192"
    v1_dir = tmp_path / "v1"
    monkeypatch.setattr(tuning, "_TABLE_DIRS", [v1_dir])
    monkeypatch.setattr(tuning, "_EXTRA_DIRS", [])
    monkeypatch.setattr(tuning, "current_chip", lambda: chip)
    _write_routed_window_table(
        v1_dir, chip, {2: "unfolded", 3: "unfolded", 4: "unfolded"}
    )
    tuning.reset_cache()
    try:
        block = _block(7, bits=4, experts=192)
        plan = _plan(block, _inputs(1, 0))
        assert plan is not None and plan.window is not None
        assert plan.launches.window_coalesce_modes == ("none",) * 5
        assert not plan.launches.window_coalesced
        assert plan.launches.window_coalesced_kernel is None
        assert plan.launches.window_topk_coalesced_kernel is None
        routed.warm_moe_model(block, mtp_enabled=True)
        assert (4, 192, False, "unfolded") not in plan.window.ran
    finally:
        tuning.reset_cache()


def test_coalesce_mode_declines_folded_when_unfolded_call():
    """pre-p3-d32d3543 surviving mutant: folded mode returns 'none' when
    folded=False (e.g. a non-folded block or M > _TOPK_FOLD_MAX_ROWS)."""
    block = _block(11, bits=4)
    plan = _plan(block, _inputs(1, 0))
    assert plan is not None
    launches = plan.launches._replace(
        window_coalesce_modes=("none", "none", "folded", "none", "unfolded")
    )
    assert routed._coalesce_mode(launches, 2, True, True) == "folded"
    assert routed._coalesce_mode(launches, 2, False, True) == "none"
    assert routed._coalesce_mode(launches, 2, True, False) == "none"


def _write_routed_down_table(v1_dir, chip, policies_by_bucket, *, bits=4, group=64):
    import json

    from omlx.formats import tuning

    routed._ensure_tuning_registered()
    skel = tuning.skeleton_digest("routed_down")
    entries = []
    for bucket, params in policies_by_bucket.items():
        for dtype_name in ("bf16", "fp16"):
            entries.append(
                {
                    "key": {
                        "route": "routed_down",
                        "face_id": "*:lut_small",
                        "dtype": dtype_name,
                        "bits": bits,
                        "group": group,
                        "rows_bucket": bucket,
                        "n": HIDDEN,
                        "k": INTER,
                    },
                    "face_sha256": "",
                    "skeleton_sha256": skel,
                    "measured_at": "2026-11-25T01:15:00Z",
                    "exactness": "bitwise",
                    "params": params,
                }
            )
    v1_dir.mkdir(parents=True, exist_ok=True)
    (v1_dir / f"{chip}.json").write_text(
        json.dumps({"schema_version": 1, "chip": chip, "entries": entries})
    )


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "fp16"])
def test_routed_down_split_and_fused_rps8_bitwise_and_dispatch(
    dtype, monkeypatch, tmp_path
):
    """Candidate 3: per-chip DATA selects 2-SG split down+shared+combine (M5)
    or fused rps=8 (M3), both 100% bitwise equal to serial 1-row decode across
    even/odd M and bf16/fp16; unused kernels are never built, proofs separate
    policies, and emitted Metal has zero uncalled helpers or hand tables."""
    from omlx.formats import tuning
    from tests.test_formats_affine import uncalled
    from tests.test_formats_tables import tabled_dot
    from tests.test_metal_tables import hand_tables

    tabled_dot(monkeypatch)
    tag = "bf16" if dtype == mx.bfloat16 else "fp16"
    v1_dir = tmp_path / "v1"
    chip_split = f"applegpu_g17s_c40_c3_{tag}"
    chip_fused8 = f"applegpu_g15d_c80_c3_{tag}"
    chip_unknown = f"applegpu_g99z_c1_c3_{tag}"
    monkeypatch.setattr(tuning, "_TABLE_DIRS", [v1_dir])
    monkeypatch.setattr(tuning, "_EXTRA_DIRS", [])
    _write_routed_down_table(
        v1_dir,
        chip_split,
        {
            "4": {"mode": "split", "rps": 4, "unroll_pairs": True},
            "5": {"mode": "split", "rps": 8, "unroll_pairs": True},
            "6-8": {"mode": "split", "rps": 8, "unroll_pairs": True},
            "9-12": {"mode": "split", "rps": 8, "unroll_pairs": True},
            "13-16": {"mode": "split", "rps": 8, "unroll_pairs": True},
        },
    )
    _write_routed_down_table(
        v1_dir,
        chip_fused8,
        {
            "6-8": {"mode": "fused", "rps": 8, "unroll_pairs": False},
            "9-12": {"mode": "fused", "rps": 8, "unroll_pairs": False},
            "13-16": {"mode": "fused", "rps": 8, "unroll_pairs": False},
        },
    )
    active = [chip_split]
    monkeypatch.setattr(tuning, "current_chip", lambda: active[0])
    tuning.reset_cache()

    built: dict[str, tuple[str, str]] = {}
    orig_mk = mx.fast.metal_kernel

    def spy_mk(**spec):
        built[spec["name"]] = (spec.get("header", ""), spec["source"])
        return orig_mk(**spec)

    monkeypatch.setattr(mx.fast, "metal_kernel", spy_mk)
    for builder in (
        routed._down_window_routed_rows_kernel,
        routed._down_window_shared_rows_kernel,
        routed._down_window_combine_kernel,
    ):
        builder.cache_clear()

    split_calls: list[tuple[int, int, bool]] = []
    fused_calls: list[tuple[int, int, bool]] = []
    orig_split = routed._window_down_split
    orig_fused = routed._window_down_fused

    def spy_split(launches, policy, h, indices, scores, down, shared_down):
        split_calls.append((int(indices.shape[0]), policy.rps, policy.unroll_pairs))
        return orig_split(launches, policy, h, indices, scores, down, shared_down)

    def spy_fused(launches, policy, h, indices, scores, down, shared_down):
        fused_calls.append((int(indices.shape[0]), policy.rps, policy.unroll_pairs))
        return orig_fused(launches, policy, h, indices, scores, down, shared_down)

    monkeypatch.setattr(routed, "_window_down_split", spy_split)
    monkeypatch.setattr(routed, "_window_down_fused", spy_fused)

    try:
        block = _block(13, bits=4, dtype=dtype)
        x1 = _inputs(1, 0, dtype=dtype)

        # 1. Split table (M5-like): builds split unrolled + fused unrolled ONLY.
        plan_split = _plan(block, x1)
        assert plan_split is not None and plan_split.window is not None
        ls = plan_split.launches
        assert ls.window_down_kernel is None
        assert ls.window_down_unrolled_kernel is not None
        assert ls.window_down_routed_kernel is None
        assert ls.window_down_routed_unrolled_kernel is not None
        assert ls.window_down_shared_kernel is not None
        assert ls.window_down_combine_kernel is not None

        for rows, shape in (
            (2, (2, 1, HIDDEN)),
            (4, (4, 1, HIDDEN)),
            (5, (1, 5, HIDDEN)),
            (8, (4, 2, HIDDEN)),
            (9, (1, 9, HIDDEN)),
            (12, (4, 3, HIDDEN)),
            (16, (4, 4, HIDDEN)),
        ):
            x = _inputs(rows, 50 + rows, dtype=dtype).reshape(shape)
            ref = _serial(block, x).reshape(rows, HIDDEN)
            split_calls.clear()
            fused_calls.clear()
            got = routed.routed_verify_window(block, x)
            assert got is not None
            mx.eval(got, ref)
            assert _same_bits(got, ref), (dtype, rows, shape)
            if rows < 4:
                assert fused_calls and not split_calls, (rows, fused_calls, split_calls)
            else:
                want_rps = 4 if rows == 4 else 8
                assert split_calls and not fused_calls, (rows, split_calls, fused_calls)
                assert all(r == want_rps and u for _, r, u in split_calls), split_calls

        # Verify zero uncalled helpers on the built tabled split kernels, and
        # zero hand-written tables on table-free affine builds of all 3 builders.
        new_names = [
            n
            for n in built
            if "down_window_routed" in n
            or "down_window_shared" in n
            or "down_window_combine" in n
        ]
        assert len(new_names) == 3, new_names
        for n in new_names:
            hdr, src = built[n]
            assert uncalled(hdr + src) == [], n
        built.clear()
        dn_aff = affine.face(4, 64, False, dtype)
        sh_aff = affine.face(8, 128, False, dtype)
        routed._down_window_routed_rows_kernel.__wrapped__(dn_aff, False)
        routed._down_window_routed_rows_kernel.__wrapped__(dn_aff, True)
        routed._down_window_shared_rows_kernel.__wrapped__(sh_aff)
        routed._down_window_combine_kernel.__wrapped__()
        assert len(built) == 4, list(built)
        for n, (hdr, src) in built.items():
            assert uncalled(hdr + src) == [], n
            assert not hand_tables(hdr), n
            assert not hand_tables(src), n

        # 2. Fused rps=8 table (M3-like): builds NO split kernels, distinct proof.
        active[0] = chip_fused8
        tuning.reset_cache()
        block.__dict__.pop("_omlx_routed_decode_plan", None)
        plan_f8 = _plan(block, x1)
        assert plan_f8 is not None and plan_f8.window is not None
        lf = plan_f8.launches
        assert lf.window_down_kernel is not None
        assert lf.window_down_unrolled_kernel is not None
        assert lf.window_down_routed_kernel is None
        assert lf.window_down_routed_unrolled_kernel is None
        assert lf.window_down_shared_kernel is None
        assert lf.window_down_combine_kernel is None
        assert plan_f8.window is not plan_split.window

        fused_launch_log: list[tuple[str, int]] = []
        orig_u0_k = lf.window_down_kernel
        orig_u2_k = lf.window_down_unrolled_kernel

        def spy_k(tag, fn):
            def wrapped(**kw):
                rps_val = dict(kw["template"])["RPS"]
                fused_launch_log.append((tag, rps_val))
                return fn(**kw)

            return wrapped

        spied_lf = lf._replace(
            window_down_kernel=spy_k("u0", orig_u0_k),
            window_down_unrolled_kernel=spy_k("u2", orig_u2_k),
        )
        old_entry = block.__dict__["_omlx_routed_decode_plan"]
        block.__dict__["_omlx_routed_decode_plan"] = (
            *old_entry[:-1],
            plan_f8._replace(launches=spied_lf),
        )

        for rows, shape in (
            (4, (4, 1, HIDDEN)),
            (6, (1, 6, HIDDEN)),
            (8, (4, 2, HIDDEN)),
            (16, (4, 4, HIDDEN)),
        ):
            x = _inputs(rows, 80 + rows, dtype=dtype).reshape(shape)
            ref = _serial(block, x).reshape(rows, HIDDEN)
            split_calls.clear()
            fused_calls.clear()
            fused_launch_log.clear()
            got = routed.routed_verify_window(block, x)
            assert got is not None
            mx.eval(got, ref)
            assert _same_bits(got, ref), (dtype, rows, shape)
            assert fused_calls and not split_calls
            want_rps = 4 if rows < 6 else 8
            want_u2 = rows < 6
            want_tag = "u2" if want_u2 else "u0"
            assert all(
                r == want_rps and u is want_u2 for _, r, u in fused_calls
            ), fused_calls
            assert fused_launch_log and all(
                t == want_tag and r == want_rps for t, r in fused_launch_log
            ), fused_launch_log

        # 3. Unknown chip: parity default (fused rps=4 unrolled, no split, no rolled).
        active[0] = chip_unknown
        tuning.reset_cache()
        block.__dict__.pop("_omlx_routed_decode_plan", None)
        plan_unk = _plan(block, x1)
        assert plan_unk is not None
        lu = plan_unk.launches
        assert lu.window_down_kernel is None
        assert lu.window_down_unrolled_kernel is not None
        assert lu.window_down_routed_kernel is None
        assert lu.window_down_routed_unrolled_kernel is None
        assert lu.window_down_shared_kernel is None
        assert lu.window_down_combine_kernel is None
    finally:
        tuning.reset_cache()


def test_routed_down_split_declines_when_shared_face_has_no_vpt(monkeypatch, tmp_path):
    """When shared_expert.down_proj's Dot face has vpt == 0 (no pair/shared
    contract), split mode falls back to fused instead of crashing."""
    from dataclasses import replace

    from omlx.formats import tuning
    from tests.test_formats_tables import tabled_dot

    tabled_dot(monkeypatch)
    v1_dir = tmp_path / "v1"
    chip = "applegpu_g17s_c40_novpt_sh"
    monkeypatch.setattr(tuning, "_TABLE_DIRS", [v1_dir])
    monkeypatch.setattr(tuning, "_EXTRA_DIRS", [])
    monkeypatch.setattr(tuning, "current_chip", lambda: chip)
    _write_routed_down_table(
        v1_dir, chip, {"4": {"mode": "split", "rps": 4, "unroll_pairs": True}}
    )
    tuning.reset_cache()
    orig_proj = routed._projection_dot

    def no_vpt_shared_down(layer, k, n):
        face = orig_proj(layer, k, n)
        if face is not None and k == INTER and n == HIDDEN and not face.tables:
            return replace(face, vpt=0)
        return face

    monkeypatch.setattr(routed, "_projection_dot", no_vpt_shared_down)
    try:
        block = _block(15, bits=4)
        plan = _plan(block, _inputs(1, 0))
        assert plan is not None
        assert plan.launches.window_down_policies[4].mode == "fused"
        assert plan.launches.window_down_shared_kernel is None
    finally:
        tuning.reset_cache()


def test_transformed_format_router_uses_row_exact_and_verify_window(
    monkeypatch, engaged
):
    """pre-oqh-2c708f788 MEDIUM #1: a transformed FormatLinear router
    (_TransformedFormatLinear) must be accepted by _window_router_ok, engage
    routed_verify_window bit-identically to 1-row decode, and route through
    row_exact_linear (not self._linear) on composed verify."""
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

    from omlx.formats import oqh, testing
    from omlx.formats.linear import FormatLinear, _TransformedFormatLinear
    from omlx.patches.qwen35_moe_routed_decode import _window_router_ok
    from tests.test_formats_oqh import _group_case

    hidden, experts = 512, 64
    block = _block(
        91,
        bits=4,
        experts=experts,
        hidden=hidden,
        inter=256,
        top_k=8,
        shared_bits=4,
        shared_group=64,
        router_bits=8,
    )
    tensors, _ = _group_case(hidden, experts, 4, 128, seed=91)
    with testing.verified("example"):
        gate = FormatLinear(oqh.OQH_AFFINE, 4, 128, hidden, tensors)
    assert type(gate) is _TransformedFormatLinear
    block.gate = gate
    block.eval()
    mx.eval(block.parameters())

    ok = _window_router_ok(block.gate, mx.bfloat16)

    x = _inputs(4, 0, hidden=hidden)
    linear_calls = []
    orig_linear = Qwen3_5BatchInvariantForward._linear

    def spy_linear(self, layer, inp):
        linear_calls.append(layer)
        return orig_linear(self, layer, inp)

    monkeypatch.setattr(Qwen3_5BatchInvariantForward, "_linear", spy_linear)
    out = _verify(block, x, window=False, row_exact=True)
    mx.eval(out)
    gate_via_linear = any(layer is block.gate for layer in linear_calls)
    assert (
        ok and not gate_via_linear
    ), f"window_router_ok={ok}, gate_via_linear={gate_via_linear}"

    monkeypatch.setattr(Qwen3_5BatchInvariantForward, "_linear", orig_linear)
    # Fused verify window engages and matches 1-row serial decode and composed
    # verify bit-for-bit.
    _check(block, x, engaged, row_exact=True)

    # Biased transformed router still declines _window_router_ok.
    block.gate["bias"] = mx.zeros((experts,), dtype=mx.bfloat16)
    assert not _window_router_ok(block.gate, mx.bfloat16)
    _BLOCKS.clear()
