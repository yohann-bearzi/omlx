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
):
    """A real-shape oQ block: quantized routed experts, 8-bit gs128 shared
    expert, 8-bit gs64 shared-expert gate, ``dtype`` router."""
    key = (seed, bits, experts, hidden, inter, top_k, dtype)
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
            shared, name, nn.QuantizedLinear.from_linear(getattr(shared, name), 128, 8)
        )
    block.shared_expert_gate = nn.QuantizedLinear.from_linear(
        block.shared_expert_gate, 64, 8
    )
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


def _verify(block, x, window):
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

    fused = routed.routed_verify_window
    if not window:
        routed.routed_verify_window = lambda block, x: None
    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=True)
    try:
        out = Qwen3_5BatchInvariantForward()._feed_forward(block, x)
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)
        routed.routed_verify_window = fused
    mx.eval(out)
    return out


def _check(block, x, engaged):
    """Window rows == one-token rows == the composed verifier's rows (the
    latter only up to the shared router cap: past it the composed verifier
    runs stock argpartition routing, not the one-token arithmetic)."""
    count = len(engaged)
    new = _verify(block, x, window=True)
    ref = _serial(block, x)
    old = _verify(block, x, window=False)
    mx.eval(ref, old)
    assert engaged[count:] == [True]
    assert _same_bits(new, ref)
    if x.size // x.shape[-1] <= router._MAX_ROWS:
        assert _same_bits(new, old)


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
    window_rows = (1, 2, 3, 4, 5, 8, 9, 16) if topk_fold else (1, 2, 3, 4)
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


@pytest.mark.parametrize("rows", [2, 5, 8])
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
            for rows in (3, 7):
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
    x = _inputs(4, 1)
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


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_top8_window_and_batched_rows_equal_one_token_decode(dtype, engaged):
    """Qwen3.5/3.6-35B-A3B oQ layout (hidden 2048, intermediate 512, top-8):
    verify windows and batched decode rows equal the fused one-token decode
    of each row."""
    hidden = 2048
    block = _block(4, bits=4, hidden=hidden, inter=512, top_k=8, dtype=dtype)
    for rows in (1, 3, 8, 16):
        for step in range(3):
            _check(block, _inputs(rows, step, hidden=hidden, dtype=dtype), engaged)
    x = _inputs(5, 0, hidden=hidden, dtype=dtype).reshape(5, 1, hidden)
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
    f32, top_k, experts, rows = mx.float32, 10, 64, 4
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
    probe = (mx.random.normal((rows, EXPERTS)) * 3).astype(mx.bfloat16)
    mx.eval(router.softmax_topk_row(probe[:1], 10), router.softmax_topk_rows(probe, 10))
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
def test_coalesced_window_kernels_equal_one_token_decode(dtype, rows, monkeypatch):
    """Pair-coalesced verify-window kernels (3-launch folded and 4-launch
    meta-driven, decoding each shared expert block once for two rows via the
    dot face's decode/apply contract) and the unrolled window down kernel
    reproduce one-token decode bit for bit across dtypes and overlap regimes."""
    # The coalesced path ships for the measured shape only (Flash-Next's
    # windows); the other activation type takes the same shape, and a shape
    # outside the allowlist is covered by its own gating test.
    from tests.test_formats_tables import tabled_dot

    tabled_dot(monkeypatch)
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


def test_coalesced_window_kernels_engage_for_tabled_face_and_have_no_dead_code(
    monkeypatch, engaged
):
    """A tabled dot face engages the coalesced window kernels at M=2 and M=4
    (matching one-token serial decode bit for bit), and the new builders carry
    no uncalled Metal functions or hand-written lookup tables."""
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
    block = _block(0, bits=4)
    for rows in (2, 3, 4):
        for step in range(3):
            _check(block, _inputs(rows, step), engaged)
    names = list(built)
    assert any("gate_up_window_coalesced" in n for n in names), names
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
