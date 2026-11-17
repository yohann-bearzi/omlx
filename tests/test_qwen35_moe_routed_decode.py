# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness and routing tests for the fused one-token routed experts."""

from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

import omlx.patches.qwen35_moe_routed_decode as routed
from omlx import formats
from omlx.formats import affine, kernels, linear
from omlx.patches.module_cache import cached_per_module
from tests.test_formats_affine import uncalled

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

EXPERTS = 32


class _FakeQwen4Model:
    pass


_FakeQwen4Model.__module__ = "mlx_vlm.models.qwen4_exp.qwen4_exp"


@pytest.fixture(autouse=True)
def _patched_block():
    """Apply router + routed patches, restore the class afterwards."""
    from mlx_vlm.models.qwen3_5_moe import language as vlm_moe

    from omlx.patches.qwen35_moe_router import apply_qwen35_moe_router_patch

    cls = vlm_moe.Qwen3_5MoeSparseMoeBlock
    apply_qwen35_moe_router_patch()  # process-wide and idempotent
    assert cls._omlx_router_fused
    call = cls.__call__
    original = getattr(call, "_omlx_routed_decode_original", call)
    cls.__call__ = original
    if "_omlx_routed_decode" in cls.__dict__:
        delattr(cls, "_omlx_routed_decode")
    assert routed.apply_qwen35_moe_routed_decode_patch()
    yield cls
    cls.__call__ = original
    cls._omlx_router_fused = True
    if "_omlx_routed_decode" in cls.__dict__:
        delattr(cls, "_omlx_routed_decode")


def _block(
    hidden,
    inter,
    top_k=10,
    bits=4,
    group_size=64,
    seed=0,
    experts=EXPERTS,
    quantized_shared=True,
    dtype=mx.bfloat16,
    shared_inter=None,
):
    """A block laid out like Qwen3.8-Flash-Next oQ: quantized routed experts,
    8-bit shared expert (gs128 where the shape allows) as wide as the routed
    ones unless ``shared_inter`` says, 8-bit gs64 shared-expert gate,
    ``dtype`` router. ``quantized_shared=False`` keeps the shared expert and
    its gate in ``dtype``."""
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    from omlx.patches.qwen35_moe_gate_up import apply_qwen35_moe_gate_up_fusion

    mx.random.seed(seed)
    shared_inter = shared_inter or inter
    args = SimpleNamespace(
        hidden_size=hidden,
        moe_intermediate_size=inter,
        shared_expert_intermediate_size=shared_inter,
        num_experts=experts,
        num_experts_per_tok=top_k,
        norm_topk_prob=True,
    )
    block = Qwen3_5MoeSparseMoeBlock(args)
    block.set_dtype(dtype)
    sm = block.switch_mlp
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(sm, name, getattr(sm, name).to_quantized(group_size, bits))
    if quantized_shared:
        shared = block.shared_expert
        shared_gs = 128 if hidden % 128 == 0 and shared_inter % 128 == 0 else 64
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(
                shared, name, nn.QuantizedLinear.from_linear(getattr(shared, name), shared_gs, 8)
            )
        block.shared_expert_gate = nn.QuantizedLinear.from_linear(block.shared_expert_gate, 64, 8)
    block.eval()
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("mlp.switch_mlp", sm)]
    assert apply_qwen35_moe_gate_up_fusion(model) == 1
    mx.eval(block.parameters())
    return block


def _pair(block, x):
    """The composed body's output and the patched call's."""
    ref = type(block).__call__._omlx_routed_decode_original(block, x)
    mx.eval(ref)
    out = block(x)
    mx.eval(out)
    return ref, out


def _plan(block, x):
    """The plan that one token ``x`` runs through, as the block's call finds
    it, or None."""
    if x.size != x.shape[-1]:
        return None
    plan = cached_per_module(
        block, "_omlx_routed_decode_plan", routed._build_plan, depth=2
    )
    return plan if plan is not None and routed._routes(plan, x) else None


def _compiled(monkeypatch):
    """The functions that MLX's compile is given, as it is given them: each
    with the calls it then serves and the threads that call it."""
    compile_, compiled = mx.compile, []

    def recorded(fn, *args, **kwargs):
        graph = compile_(fn, *args, **kwargs)
        record = SimpleNamespace(calls=0, threads=set())
        compiled.append(record)

        def called(*arrays):
            record.calls += 1
            record.threads.add(threading.get_ident())
            return graph(*arrays)

        return called

    monkeypatch.setattr(mx, "compile", recorded)
    return compiled


# The kernels of a block's launches: the routed decode's and the router's.
_BLOCK_KERNELS = ("omlx_qwen35_moe_",)
_BUILDERS = (
    "_gate_up_kernel",
    "_down_kernel",
    "_gate_up_window_kernel",
    "_down_window_kernel",
    "_gate_up_topk_kernel",
)
_ROUTER_KERNELS = (
    "_GEMV_KERNEL",
    "_SOFTMAX_TOPK_KERNEL",
    "_SOFTMAX_TOPK_ROWS_KERNEL",
    "_KERNEL",
    "_COMBINE_KERNEL",
)


@pytest.fixture
def launched(monkeypatch):
    """Every launch of a block's kernels, as it is built: the kernel's
    name, its inputs and outputs (``launched.built``), while
    ``launched.recording``. ``launched.verbose = True`` has MLX print the
    source it generates for each."""
    from omlx.patches import qwen35_moe_router as router

    make, built = mx.fast.metal_kernel, []
    built_with = SimpleNamespace(verbose=False, recording=True)

    def kernel(**spec):
        made = make(**spec)
        if not spec["name"].startswith(_BLOCK_KERNELS):
            return made

        def launch(**call):
            if not built_with.recording:
                return made(**call)
            outputs = made(**call, verbose=built_with.verbose)
            built.append(
                SimpleNamespace(
                    name=spec["name"], inputs=call["inputs"], outputs=outputs
                )
            )
            return outputs

        return launch

    monkeypatch.setattr(mx.fast, "metal_kernel", kernel)
    for name in _ROUTER_KERNELS:
        monkeypatch.setattr(router, name, None)
    for name in _BUILDERS:
        getattr(routed, name).cache_clear()
    built_with.built = built
    yield built_with
    for name in _BUILDERS:
        getattr(routed, name).cache_clear()


def _engaged():
    """Whether one token's launches ran for every plan built since the test
    began, and no window's failed."""
    tokens, windows = (
        [proof for (kernel, *_), proof in kernels._PROOFS.items() if kernel == name]
        for name in ("routed_decode", "routed_window")
    )
    return (
        bool(tokens)
        and all(proof.ran and not proof.failed for proof in tokens)
        and not any(proof.failed for proof in windows)
    )


def _logged(caplog, module, level=logging.DEBUG):
    """What ``module`` logged. Other loggers are at whatever level the tests
    before this one left them."""
    return [
        r for r in caplog.records if r.name == module.__name__ and r.levelno >= level
    ]


def _same_bits(a, b):
    return a.shape == b.shape and mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()


def _does_not_compile(a):
    """An array like ``a`` that fails when it is evaluated: MLX compiles a
    kernel then."""
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
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
    )[0]


class _Counted(nn.Module):
    """A module that records its calls."""

    def __init__(self, module, calls):
        super().__init__()
        self.module = module
        self.calls = calls

    def __call__(self, x):
        self.calls.append(1)
        return self.module(x)


class _CastGate(nn.Module):
    """A router whose logits are not in its input's dtype."""

    def __init__(self, gate, dtype):
        super().__init__()
        self.gate = gate
        self.dtype = dtype

    def __call__(self, x):
        return self.gate(x).astype(self.dtype)


@pytest.mark.parametrize(
    "hidden,inter,bits,group_size",
    [
        (2560, 640, 5, 64),  # Qwen3.8-Flash-Next oQ5e
        (2560, 640, 4, 64),
        (2560, 640, 3, 64),  # Qwen3.8-Flash-Next oQ3e
        (2560, 640, 2, 64),
        (1024, 320, 3, 32),
        (1024, 384, 2, 128),
        (1024, 320, 5, 32),
        (1024, 384, 6, 128),
        (1024, 320, 8, 64),
        # Gate+up takes qmv: K ends in a partial kernel block, or is whole
        # blocks of qmv's but not of qmv_fast's.
        (960, 320, 4, 64),
        (1280, 320, 5, 64),
        (896, 320, 6, 64),
        (832, 320, 8, 64),
    ],
)
def test_fused_decode_is_bit_identical(hidden, inter, bits, group_size, monkeypatch):
    compiled = _compiled(monkeypatch)
    for seed in range(2):
        block = _block(hidden, inter, bits=bits, group_size=group_size, seed=seed)
        for step in range(6):
            x = (mx.random.normal((1, 1, hidden)) * (0.5 + step)).astype(mx.bfloat16)
            # The shared expert and its gate run inside the two launches.
            assert _plan(block, x).launches.fold
            ref, out = _pair(block, x)
            assert _same_bits(ref, out)
    # Every call is the one graph's: the blocks share a proof.
    assert [graph.calls for graph in compiled] == [12]
    assert _engaged()


@pytest.mark.parametrize("experts", [EXPERTS, 128])
def test_bf16_shared_expert_stays_composed_and_bit_identical(experts):
    # 128 experts select inside the gate+up launch, 32 in the routing launch.
    # The 128-expert block keeps hidden < 16 x experts, so the fused router gemv runs.
    hidden, inter = (2560, 640) if experts == EXPERTS else (1024, 320)
    block = _block(hidden, inter, bits=5, quantized_shared=False, experts=experts)
    for step in range(4):
        x = (mx.random.normal((1, 1, hidden)) * (0.5 + step)).astype(mx.bfloat16)
        plan = _plan(block, x)
        assert not plan.launches.fold
        assert routed._topk_folds(plan.launches, block.gate(x)) == (experts == 128)
        ref, out = _pair(block, x)
        assert _same_bits(ref, out)
    assert _engaged()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("quantized_shared", [True, False])
@pytest.mark.parametrize(
    "hidden,inter,bits,group_size",
    [(960, 320, 4, 32), (896, 384, 5, 128), (832, 320, 6, 32), (896, 384, 8, 128)],
)
def test_qmv_gate_up_is_bit_identical(
    hidden, inter, bits, group_size, quantized_shared, dtype
):
    """K ends in a partial kernel block: gate+up, the shared expert's and the
    gate row take ``qmv``."""
    block = _block(
        hidden,
        inter,
        bits=bits,
        group_size=group_size,
        quantized_shared=quantized_shared,
        dtype=dtype,
    )
    for step in range(4):
        x = (mx.random.normal((1, 1, hidden)) * (0.5 + step)).astype(dtype)
        assert _plan(block, x).launches.fold == quantized_shared
        ref, out = _pair(block, x)
        assert _same_bits(ref, out)
    assert _engaged()


def test_a_launch_holds_the_swiglu_only_where_it_applies_it(monkeypatch):
    """MLX hashes a kernel's whole source at every launch: the down launches,
    which apply no SwiGLU, do not carry its function, and no launch carries
    a function it does not call."""
    built = {}

    def build(**kernel):
        built[kernel["name"]] = kernel["header"] + kernel["source"]

    monkeypatch.setattr(mx.fast, "metal_kernel", build)
    face = partial(affine.face, dtype=mx.bfloat16)
    gate_up, down, shared, gate = (
        face(4, 64, True),
        face(4, 64, False),
        face(8, 128, True),
        face(8, 64, True),
    )
    for kernel, faces in (
        (routed._gate_up_kernel, (gate_up, None, None)),
        (routed._gate_up_kernel, (gate_up, shared, gate)),
        (routed._gate_up_window_kernel, (gate_up, shared, gate)),
        (routed._gate_up_topk_kernel, (gate_up, None, None)),
        (routed._gate_up_topk_kernel, (gate_up, shared, gate)),
        (routed._down_kernel, (down, None)),
        (routed._down_kernel, (down, shared)),
        (routed._down_window_kernel, (down, shared)),
    ):
        kernel.__wrapped__(*faces)
    assert len(built) == 8
    for name, metal in built.items():
        assert ("swiglu_store" in metal) == ("_gate_up_" in name), name
        assert uncalled(metal) == [], name


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_the_folded_top_k_launch_holds_its_dtype_s_selection_alone(monkeypatch, dtype):
    """MLX hashes a launch's whole source at every launch: the bf16 kernel
    holds the packed-key selection, the others the scan, each under a name
    of its own."""
    built = {}

    def build(**kernel):
        built[kernel["name"]] = kernel["header"]

    monkeypatch.setattr(mx.fast, "metal_kernel", build)
    routed._gate_up_topk_kernel.__wrapped__(affine.face(4, 64, True, dtype), None, None)
    ((name, header),) = built.items()
    bf16 = dtype == mx.bfloat16
    assert ("struct omlx_router_topk<bfloat16_t, NE> {" in header) == bf16
    assert ("struct omlx_router_topk {" in header) == (not bf16)
    assert ("_topk_bf16_" in name) == bf16


@pytest.mark.parametrize("hidden", [2560, 2432])
@pytest.mark.parametrize("bits", [2, 3, 4, 5])
def test_fp32_kernels_match_mlx_mat_vecs(bits, hidden):
    """BF16 outputs hide one-ulp FP32 differences, so run both launches in
    FP32 against MLX's FP32 mat-vecs: the gate+up rows after SwiGLU of the
    experts and the shared expert plus the gate row, then every down row
    read through the combine (a one-hot score with the gate at sigmoid 0
    reads one expert; zero scores with the gate at sigmoid 1 the shared one).
    Hidden 2560 is whole kernel blocks (``qmv_fast``), 2432 is not (``qmv``)."""
    from mlx_vlm.models.activations import swiglu

    inter, gs, top_k, fast = 640, 64, 10, hidden == 2560
    f32 = mx.float32
    mx.random.seed(40 + bits)

    def quantized(shape, group_size, b):
        return mx.quantize(mx.random.normal(shape) * 0.05, group_size, b)

    def qmm(x, weights, group_size, b):
        return mx.quantized_matmul(x, *weights, transpose=True, group_size=group_size, bits=b)

    experts_gate_up = quantized((EXPERTS, 2 * inter, hidden), gs, bits)
    experts_down = quantized((EXPERTS, hidden, inter), gs, bits)
    shared_gate, shared_up = quantized((inter, hidden), 128, 8), quantized((inter, hidden), 128, 8)
    shared_down = quantized((hidden, inter), 128, 8)
    gate_row = quantized((1, hidden), 64, 8)
    fmt = partial(affine.face, dtype=f32)
    gate_up_kernel = routed._gate_up_kernel(
        fmt(bits, gs, fast), fmt(8, 128, fast), fmt(8, 64, fast)
    )
    down_kernel = routed._down_kernel(fmt(bits, gs, False), fmt(8, 128, False))
    for step in range(3):
        x = mx.random.normal((1, 1, hidden)) * (0.5 + step)
        ids = mx.random.permutation(EXPERTS)[:top_k].astype(mx.uint32)
        routed_gate_up = mx.gather_qmm(
            mx.expand_dims(x, (-2, -3)), *experts_gate_up, rhs_indices=ids.reshape(1, 1, top_k),
            transpose=True, group_size=gs, bits=bits, sorted_indices=False,
        )
        gate, up = mx.split(routed_gate_up, 2, axis=-1)
        ref_h = mx.concatenate([
            swiglu(gate, up).reshape(-1),
            swiglu(qmm(x, shared_gate, 128, 8), qmm(x, shared_up, 128, 8)).reshape(-1),
            qmm(x, gate_row, 64, 8).reshape(-1),
        ])
        h = gate_up_kernel(
            inputs=[x, *experts_gate_up, ids, *shared_gate, *shared_up, *gate_row],
            template=[
                ("T", f32), ("K", hidden), ("NI", inter), ("RPS", 2), ("NSG", 2), ("NS", inter),
                ("TOPK", top_k),
            ],
            grid=(32, 2 * (1 + inter // 4 + top_k * inter // 4), 1),
            threadgroup=(32, 2, 1),
            output_shapes=[ref_h.shape],
            output_dtypes=[f32],
        )[0]
        assert mx.array_equal(h.view(mx.uint32), ref_h.view(mx.uint32)).item()

        ref_down = mx.gather_qmm(
            mx.expand_dims(h[: top_k * inter].reshape(top_k, inter), -2), *experts_down,
            rhs_indices=ids, transpose=True, group_size=gs, bits=bits, sorted_indices=False,
        ).reshape(top_k, hidden)
        ref_shared = qmm(h[top_k * inter : -1], shared_down, 128, 8)
        for j in range(top_k + 1):
            scores = (mx.arange(top_k) == j).astype(f32)
            saturated_gate = mx.array([-1e30 if j < top_k else 1e30], f32)
            row = down_kernel(
                inputs=[
                    mx.concatenate([h[:-1], saturated_gate]), *experts_down, *shared_down, ids,
                    scores,
                ],
                template=[
                    ("T", f32), ("K", inter), ("N", hidden), ("RPS", routed._down_rows(1)),
                    ("KS", inter), ("NPART", top_k + 1), ("TOPK", top_k),
                ],
                grid=(32, (top_k + 1) * hidden // routed._down_rows(1), 1),
                threadgroup=(32, top_k + 1, 1),
                output_shapes=[(hidden,)],
                output_dtypes=[f32],
            )[0]
            # The +0 folds of the combine turn -0 into +0, so compare values.
            assert mx.array_equal(row, ref_down[j] if j < top_k else ref_shared).item()


@pytest.mark.parametrize("bits", [4, 5])
def test_fp32_folded_routing_matches_the_routing_launch(bits):
    """The gate+up launch that selects its rows' experts itself, in FP32 (BF16
    outputs hide one-ulp differences): per row, the selection and scores of
    the routing launch, and the gate+up rows of the launch fed with them.
    Router logits repeat, so probabilities tie."""
    from omlx.patches import qwen35_moe_router as router

    hidden, inter, gs, experts, rows, top_k = 1024, 384, 64, 128, 3, 10
    f32 = mx.float32
    mx.random.seed(50 + bits)

    def quantized(shape, group_size, b):
        return mx.quantize(mx.random.normal(shape) * 0.05, group_size, b)

    experts_gate_up = quantized((experts, 2 * inter, hidden), gs, bits)
    shared = quantized((inter, hidden), 128, 8) + quantized((inter, hidden), 128, 8)
    gate_row = quantized((1, hidden), 64, 8)
    fmt = partial(affine.face, dtype=f32)
    formats = (fmt(bits, gs, True), fmt(8, 128, True), fmt(8, 64, False))
    x = mx.random.normal((rows, hidden))
    logits = mx.random.normal((rows, experts // 2)) * 2
    logits = mx.concatenate([logits, logits[:, ::-1]], axis=-1)
    template = [
        ("T", f32), ("K", hidden), ("NI", inter), ("RPS", 2), ("NSG", 2), ("NS", inter),
        ("TOPK", top_k),
    ]
    width = top_k * inter + inter + 1
    blocks = 1 + inter // 4 + top_k * inter // 4
    h, inds, scores = routed._gate_up_topk_kernel(*formats)(
        inputs=[x, *experts_gate_up, logits, *shared, *gate_row],
        template=template + [("NE", experts), ("M", rows), ("YW", width)],
        grid=(32, 2 * blocks * rows, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(rows, width), (rows, top_k), (rows, top_k)],
        output_dtypes=[f32, mx.uint32, f32],
    )
    mx.eval(router.softmax_topk_rows(logits.astype(mx.bfloat16), top_k))
    ref_inds, ref_scores = router._SOFTMAX_TOPK_ROWS_KERNEL(
        inputs=[logits],
        template=[("T", f32), ("NE", experts), ("K", top_k)],
        grid=(32, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, top_k), (rows, top_k)],
        output_dtypes=[mx.uint32, f32],
    )
    assert mx.array_equal(inds, ref_inds).item()
    assert mx.array_equal(scores.view(mx.uint32), ref_scores.view(mx.uint32)).item()
    for r in range(rows):
        ref_h = routed._gate_up_kernel(*formats)(
            inputs=[x[r], *experts_gate_up, ref_inds[r], *shared, *gate_row],
            template=template,
            grid=(32, 2 * blocks, 1),
            threadgroup=(32, 2, 1),
            output_shapes=[(width,)],
            output_dtypes=[f32],
        )[0]
        assert mx.array_equal(h[r].view(mx.uint32), ref_h.view(mx.uint32)).item()


def test_experts_past_the_bound_view_are_read_from_the_stacked_weights():
    """The kernels bind a one-expert view and index the rest; a view copied
    out of the stacked buffer would read the wrong bytes for high experts."""
    from omlx.patches.qwen35_moe_router import fused_moe_combine

    block = _block(1024, 320, bits=5, experts=128)
    plan = _plan(block, mx.zeros((1, 1, 1024), mx.bfloat16))
    launches, (_, gate_up, down, shared_gate_up, shared_down) = (
        plan.launches,
        plan.operands,
    )
    assert all(o.shape[0] == 1 for o in gate_up + down)
    for ids in (list(range(118, 128)), [127, 0, 116, 64, 1, 126, 3, 115, 7, 100]):
        x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
        inds = mx.array(ids, dtype=mx.uint32).reshape(1, 1, 10)
        scores = mx.softmax(mx.random.normal((1, 1, 10)), axis=-1).astype(mx.bfloat16)
        shared = block["shared_expert"](x)
        gate = block["shared_expert_gate"](x)
        ref = fused_moe_combine(block.switch_mlp(x, inds), scores, shared, gate)
        h = routed._gate_up(launches, x, inds, gate_up, shared_gate_up)
        out = routed._down(launches, h, inds, scores, x.shape, down, shared_down)
        mx.eval(ref, out)
        assert _same_bits(ref, out)


@pytest.mark.parametrize(
    "shape,viewed",
    [
        ((4, 8, 2), True),
        ((4, 8), True),
        # MLX binds an input of fewer than 8 elements as constants.
        ((4, 7), False),
        ((4, 4, 1), False),
        ((16,), False),
        ((), False),
    ],
)
def test_an_expert_of_fewer_than_8_elements_is_bound_with_its_stack(shape, viewed):
    a = mx.zeros(shape)
    mx.eval(a)
    view = routed._expert_view(a)
    if viewed:
        assert view.shape == (1, *shape[1:])
    else:
        assert view is a


def test_tied_router_logits_route_like_the_served_block():
    """Duplicated router rows give exactly tied probabilities; ties must pick
    the same experts, in the same order, with the same scores as the served
    router launches."""
    from omlx.patches.qwen35_moe_router import router_logits_row, softmax_topk_row

    block = _block(1024, 320, bits=5, experts=128)
    rows = block.gate.weight[:16]
    block.gate.weight = mx.concatenate([rows] * 8)[mx.random.permutation(128)]
    mx.eval(block.gate.weight)
    for step in range(8):
        x = (mx.random.normal((1, 1, 1024)) * (0.5 + step)).astype(mx.bfloat16)
        # Both router launches engage: the gemv and the softmax + top-k.
        assert router_logits_row(x, block.gate.weight) is not None
        assert softmax_topk_row(block.gate(x), 10) is not None
        ref, out = _pair(block, x)
        assert _same_bits(ref, out)



@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("top_k", [8, 10])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_fast_down_and_top8_are_bit_identical(dtype, top_k, bits):
    """Qwen3.5/3.6-35B-A3B layout: hidden 2048 and intermediate 512 put the
    down projection on ``qmv_fast`` too."""
    block = _block(1024, 512, bits=bits, top_k=top_k, dtype=dtype)
    for step in range(8):
        x = (mx.random.normal((1, 1, 1024)) * (0.5 + step)).astype(dtype)
        plan = _plan(block, x)
        launches = plan.launches
        assert launches.fold and launches.dtype == dtype and launches.top_k == top_k
        ref, out = _pair(block, x)
        assert _same_bits(ref, out)
    assert _engaged()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("quantized_shared", [True, False])
def test_top8_routing_folded_into_gate_up_is_bit_identical(dtype, quantized_shared):
    """128 experts select inside the gate+up launch; an unquantized shared
    expert runs composed and enters the combine."""
    block = _block(
        1024, 512, top_k=8, experts=128, quantized_shared=quantized_shared, dtype=dtype
    )
    for step in range(4):
        x = (mx.random.normal((1, 1, 1024)) * (0.5 + step)).astype(dtype)
        plan = _plan(block, x)
        assert plan.launches.fold == quantized_shared
        assert routed._topk_folds(plan.launches, block.gate(x))
        ref, out = _pair(block, x)
        assert _same_bits(ref, out)
    assert _engaged()


def test_mismatched_dtypes_keep_the_composed_body():
    block = _block(1024, 512, top_k=8, dtype=mx.float16)
    x = mx.zeros((1, 1, 1024), mx.float16)
    # A router of another dtype runs as the block's own linear.
    gate = block.gate["weight"]
    block.gate["weight"] = gate.astype(mx.bfloat16)
    assert _plan(block, x).launches.router is None
    block.gate["weight"] = gate
    assert _plan(block, x.astype(mx.bfloat16)) is None
    down = block.switch_mlp.down_proj
    down["scales"] = down["scales"].astype(mx.bfloat16)
    down["biases"] = down["biases"].astype(mx.bfloat16)
    assert _plan(block, x) is None


def test_router_logits_of_another_dtype_keep_the_composed_body():
    block = _block(512, 256, top_k=8, experts=128)
    x = mx.random.normal((1, 1, 512)).astype(mx.bfloat16)
    logits = block.gate(x)
    block.gate = _CastGate(block.gate, mx.float16)
    plan = _plan(block, x)
    assert plan.launches.router is None
    # The folded routing takes fp16 logits, in a launch compiled for fp16.
    assert routed._topk_folds(plan.launches, logits)
    assert not routed._topk_folds(plan.launches, block.gate(x))
    assert _same_bits(*_pair(block, x))
    assert not plan.proof.ran and not plan.proof.failed


def test_a_shared_expert_of_another_width_is_folded():
    block = _block(1024, 320, shared_inter=448)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    assert _plan(block, x).launches.fold
    assert _same_bits(*_pair(block, x))
    assert _engaged()


@pytest.mark.parametrize(
    "owner,names",
    [
        ("switch_mlp", ("gate_up_proj", "down_proj")),
        ("shared_expert", ("gate_proj", "up_proj", "down_proj")),
    ],
)
def test_plan_follows_replaced_weights(owner, names):
    block = _block(1024, 320, bits=5)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    first = _plan(block, x)
    donor = _block(1024, 320, bits=5, seed=7)
    for name in names:
        for key in ("weight", "scales", "biases"):
            block[owner][name][key] = donor[owner][name][key]
    assert _plan(block, x) is not first
    ref, out = _pair(block, x)
    assert _same_bits(ref, out)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden": 1024, "inter": 320, "top_k": 6},
    ],
)
def test_ineligible_shapes_keep_the_composed_body(kwargs):
    block = _block(**kwargs)
    hidden = kwargs["hidden"]
    x = mx.random.normal((1, 1, hidden)).astype(mx.bfloat16)
    assert _plan(block, x) is None


@pytest.mark.parametrize(
    "name,value",
    [
        # A threadgroup takes six of the 320 gate rows.
        ("_GATE_UP_ROWS", 3),
        ("_GATE_UP_SIMDGROUPS", 3),
        # Groups of 128 rows: whole in the hidden size and in gate+up's 640
        # rows, not in the experts' 320.
        ("_GATE_UP_ROWS", 64),
        # Groups of 5 rows: whole in the experts' width, not in the hidden
        # size. One token, a window, the widest window.
        ("_down_rows", lambda rows: 5 if rows == 1 else 4),
        ("_down_rows", lambda rows: 2 if rows == 1 else 5),
        ("_down_rows", lambda rows: 5 if rows == routed.WINDOW_MAX_ROWS else 2),
    ],
)
def test_rows_that_are_not_whole_launch_groups_keep_the_composed_body(
    name, value, monkeypatch
):
    """The grids divide the rows by a launch's rows per group. Affine experts
    are whole groups at every shape MLX quantizes; another format's are not
    bound to be."""
    monkeypatch.setattr(routed, name, value)
    block = _block(1024, 320)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    assert _plan(block, x) is None
    assert _same_bits(*_pair(block, x))
    assert not kernels._PROOFS


@pytest.mark.parametrize("dtypes", [(), (mx.bfloat16, mx.float16)])
def test_a_gate_up_face_names_the_one_dtype_of_the_launches(dtypes, monkeypatch):
    """The launches are compiled for the activations the gate+up face is
    exact for; a face exact for several does not say which."""
    face = affine.AffineFormat.dot

    def several(self, layer, k, n):
        found = face(self, layer, k, n)
        if found is None or layer["weight"].ndim == 2 or k != 1024:
            return found
        return replace(found, dtypes=dtypes)

    monkeypatch.setattr(affine.AffineFormat, "dot", several)
    block = _block(1024, 320)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    assert _plan(block, x) is None
    assert _same_bits(*_pair(block, x))
    assert not kernels._PROOFS


def test_prefill_verify_and_float_rows_keep_the_composed_body():
    block = _block(1024, 320)
    assert _plan(block, mx.zeros((1, 4, 1024), dtype=mx.bfloat16)) is None
    assert _plan(block, mx.zeros((2, 1, 1024), dtype=mx.bfloat16)) is None
    assert _plan(block, mx.zeros((1, 1, 1024), dtype=mx.float16)) is None
    x = mx.random.normal((1, 5, 1024)).astype(mx.bfloat16)
    ref, out = _pair(block, x)
    assert _same_bits(ref, out)


def test_block_without_gate_up_fusion_is_not_routed():
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    args = SimpleNamespace(
        hidden_size=1024,
        moe_intermediate_size=320,
        shared_expert_intermediate_size=320,
        num_experts=EXPERTS,
        num_experts_per_tok=10,
        norm_topk_prob=True,
    )
    block = Qwen3_5MoeSparseMoeBlock(args)
    block.set_dtype(mx.bfloat16)
    x = mx.zeros((1, 1, 1024), dtype=mx.bfloat16)
    assert _plan(block, x) is None


def test_a_failed_launch_keeps_its_faces_and_shapes_composed(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=routed.__name__)
    block = _block(1024, 320)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    calls = []
    down = routed._down

    def broken(*args):
        """A down launch that fails when it is first evaluated."""
        calls.append(1)
        return _does_not_compile(down(*args))

    with monkeypatch.context() as patch:
        patch.setattr(routed, "_down", broken)
        assert _same_bits(*_pair(block, x))
        assert _plan(block, x) is None
        # Another block of these faces and shapes does not try again.
        twin = _block(1024, 320, seed=1)
        assert _plan(twin, x) is None
        assert _same_bits(*_pair(twin, x))
        assert _same_bits(*_pair(block, x))
    assert len(calls) == 1
    warnings = _logged(caplog, routed, logging.WARNING)
    assert [r.getMessage() for r in warnings] == [
        "fused routed-expert decode failed; composed fallback"
    ]
    # The block's own gate runs once a call: the composed body's.
    asked = []
    gate = block["gate"]
    block["gate"] = _Counted(gate, asked)
    assert _same_bits(*_pair(block, x))
    block["gate"] = gate
    assert len(asked) == 2  # the reference's call, then the block's
    # Other shapes still route.
    other = _block(1024, 512)
    assert _plan(other, x) is not None
    assert _same_bits(*_pair(other, x))
    assert _plan(other, x).proof.ran


def test_a_failure_ahead_of_the_block_is_raised_where_its_output_is_evaluated(caplog):
    """The block proves its launches on zeros and builds them over its
    arrays: a failure in its input, its gate's or its router's weight is
    its caller's, raised where the caller evaluates the result."""
    caplog.set_level(logging.WARNING, logger=routed.__name__)
    composed, fused = _block(1024, 320), _block(1024, 320, experts=128)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    for block, metal in ((composed, (EXPERTS, False)), (fused, (128, True))):
        plan = _plan(block, x)
        y = block(_does_not_compile(x))
        with pytest.raises(RuntimeError, match="Unable to build metal library"):
            mx.eval(y)
        assert plan.proof.ran == {metal} and not plan.proof.failed
        weight = block.gate["weight"]
        block.gate["weight"] = _does_not_compile(weight)
        y = block(x)
        block.gate["weight"] = weight
        with pytest.raises(RuntimeError, match="Unable to build metal library"):
            mx.eval(y)
        assert not plan.proof.failed
    assert not _logged(caplog, routed, logging.WARNING)
    for block in (composed, fused):
        assert _same_bits(*_pair(block, x))
    assert _engaged()


def _evaluations(monkeypatch):
    """What ``mx.eval`` is given, call by call."""
    evaluate, evaluated = mx.eval, []

    def recorded(*arrays):
        evaluated.append(arrays)
        evaluate(*arrays)

    monkeypatch.setattr(mx, "eval", recorded)
    return evaluated


def test_an_unproven_block_evaluates_its_stand_in_and_a_proven_one_nothing(monkeypatch):
    """A block's one evaluation is the stand-in of launches whose Metal is
    not proven. Nothing of the call is evaluated: not x, not a tensor, not
    the result."""
    block = _block(1024, 320, experts=128)
    x, other = (mx.random.normal((1, 1, 1024)).astype(mx.bfloat16) for _ in range(2))
    _plan(block, x)  # evaluates the experts' views
    evaluated = _evaluations(monkeypatch)
    y = block(x)
    *inputs, (proving,) = evaluated
    assert len(proving) == 2 and not any(a is y for a in proving)
    # Ahead of it, the stand-in's logits, then its zeros: none of the call's.
    assert len(inputs) == 2 and not any(a is x for arrays in inputs for a in arrays)
    del evaluated[:]
    block(x), block(other), block(x[0])
    assert not evaluated


def test_a_face_that_does_not_compile_keeps_only_its_blocks_composed(
    monkeypatch, caplog
):
    """MLX compiles a kernel at its first evaluation. A face that fails
    there, after blocks of the same shapes ran with other faces, fails inside
    the call that can fall back."""
    caplog.set_level(logging.WARNING, logger=routed.__name__)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    good = _block(1024, 320)
    assert _same_bits(*_pair(good, x))
    assert _engaged()
    dot = formats.dot

    def broken(layer, k, n):
        face = dot(layer, k, n)
        if face is None or k != 320:
            return face
        source = face.source + 'static_assert(false, "does not compile");\n'
        return replace(face, key=face.key + "_broken", source=source)

    monkeypatch.setattr(formats, "dot", broken)
    bad = _block(1024, 320, seed=1)
    assert _same_bits(*_pair(bad, x))
    # The warning carries the compiler's message.
    (warning,) = _logged(caplog, routed, logging.WARNING)
    assert "does not compile" in str(warning.exc_info[1])
    assert _plan(bad, x) is None
    assert _plan(good, x) is not None
    assert _same_bits(*_pair(good, x))


def test_a_format_whose_dot_raises_keeps_its_blocks_composed(
    monkeypatch, caplog, dot_defects
):
    asked = []

    def raises(self, layer, k, n):
        asked.append(layer)
        raise RuntimeError("no face today")

    monkeypatch.setattr(affine.AffineFormat, "dot", raises)
    block = _block(1024, 512)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    with caplog.at_level(logging.ERROR, logger=formats.__name__):
        for _ in range(2):
            assert _same_bits(*_pair(block, x))
    assert _plan(block, x) is None
    assert not kernels._PROOFS
    # The answer is kept with the block's plan: no projection is asked twice.
    own = {id(block.switch_mlp.gate_up_proj), id(block.switch_mlp.down_proj)}
    assert asked and {id(layer) for layer in asked} <= own
    assert len({id(layer) for layer in asked}) == len(asked)
    defect = "RuntimeError('no face today')"
    assert [repr(r.exc_info[1]) for r in _logged(caplog, formats)] == [defect]
    assert dot_defects == {("affine", defect)}
    dot_defects.clear()  # expected


def test_blocks_share_a_proof_when_they_compile_the_same_launches(monkeypatch):
    def unfolded(hidden=1024, inter=320, **kwargs):
        return _block(hidden, inter, quantized_shared=False, **kwargs)

    def proofs(block):
        """One token's proof, then a window's: only a folded plan has window
        launches."""
        x = mx.zeros((1, 1, block.switch_mlp.down_proj.weight.shape[1]), mx.bfloat16)
        plan = _plan(block, x)
        if not plan.launches.fold:
            assert plan.window is None
            return (plan.proof,)
        assert plan.window is not plan.proof
        return plan.proof, plan.window

    def shared(a, b):
        return all(p is q for p, q in zip(a, b, strict=True))

    def apart(a, b):
        return not any(p is q for p, q in zip(a, b, strict=True))

    base, folded = proofs(unfolded()), proofs(_block(1024, 320))
    assert shared(proofs(unfolded(seed=1)), base)
    assert shared(proofs(_block(1024, 320, seed=1)), folded)
    assert apart(folded[:1], base)
    # One shape apart, with the same faces.
    assert apart(proofs(unfolded(hidden=2048)), base)
    assert apart(proofs(unfolded(inter=448)), base)
    assert apart(proofs(unfolded(top_k=8)), base)
    assert apart(proofs(_block(1024, 320, shared_inter=448)), folded)
    # With 128 experts the router is the fused gemv.
    assert apart(proofs(unfolded(experts=128)), base)
    dot = formats.dot

    def patched(face_of):
        monkeypatch.setattr(
            formats, "dot", lambda layer, k, n: face_of(layer, dot(layer, k, n))
        )

    def renamed(block, *layers):
        other = {id(layer) for layer in layers}

        def face_of(layer, face):
            return replace(face, key="other") if id(layer) in other else face

        patched(face_of)
        return proofs(block)

    # One face apart.
    for name in ("gate_up_proj", "down_proj"):
        block = unfolded(seed=2)
        assert apart(renamed(block, block.switch_mlp[name]), base)
    for names in (("gate_proj", "up_proj"), ("down_proj",)):
        block = _block(1024, 320, seed=2)
        layers = (block.shared_expert[name] for name in names)
        assert apart(renamed(block, *layers), folded)
    block = _block(1024, 320, seed=2)
    assert apart(renamed(block, block.shared_expert_gate), folded)

    def recast(block, *layers):
        for layer in layers:
            layer.scales = layer.scales.astype(mx.float16)
            layer.biases = layer.biases.astype(mx.float16)
        patched(lambda layer, face: replace(face, dtypes=(mx.bfloat16,)))
        return proofs(block)

    # The same faces, with one matrix's tensors in another dtype.
    for name in ("gate_up_proj", "down_proj"):
        block = unfolded(seed=2)
        assert apart(recast(block, block.switch_mlp[name]), base)
    for names in (("gate_proj", "up_proj"), ("down_proj",)):
        block = _block(1024, 320, seed=2)
        layers = (block.shared_expert[name] for name in names)
        assert apart(recast(block, *layers), folded)
    block = _block(1024, 320, seed=2)
    assert apart(recast(block, block.shared_expert_gate), folded)


def test_a_window_that_fails_keeps_its_twin_s_windows_composed(monkeypatch):
    """Blocks of the same faces and shapes share a window's proof as they
    share one token's: the twin does not build the launch again."""
    first, twin = (_block(512, 256, top_k=8, experts=128, seed=seed) for seed in (0, 1))
    x = mx.random.normal((1, 3, 512)).astype(mx.bfloat16)
    built = []

    def broken(launches, h, indices, scores, *operands):
        built.append(launches)
        rows = mx.zeros((indices.shape[0], launches.hidden), launches.dtype)
        return _does_not_compile(rows)

    monkeypatch.setattr(routed, "_window_down", broken)
    for block in (first, twin, first):
        assert routed.routed_verify_window(block, x) is None
    assert len(built) == 1
    # One token still takes the fused launches, in both.
    for block in (first, twin):
        assert _same_bits(*_pair(block, x[:, :1]))
        plan = _plan(block, x[:, :1])
        assert plan.window.failed and not plan.window.ran
        assert plan.proof.ran == {(128, True)} and not plan.proof.failed


def test_a_plan_that_folds_no_shared_expert_has_no_window():
    block = _block(512, 256, top_k=8, experts=128, quantized_shared=False)
    x = mx.random.normal((1, 3, 512)).astype(mx.bfloat16)
    assert routed.routed_verify_window(block, x) is None
    assert [kernel for kernel, *_ in kernels._PROOFS] == ["routed_decode"]


def test_a_launch_is_evaluated_once_per_router_width_and_fold(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=routed.__name__)
    x = mx.random.normal((1, 1, 512)).astype(mx.bfloat16)
    ran = set()
    for experts in (128, 256, 128):
        block = _block(512, 256, top_k=8, experts=experts)
        for _ in range(2):
            assert _same_bits(*_pair(block, x))
        ran.add((experts, True))
        plan = _plan(block, x)
        assert plan.launches.fold and plan.proof.ran == ran
    # Whether the routing folds is decided where the launches are built,
    # once for a signature: x of another shape.
    monkeypatch.setattr(routed, "_topk_folds", lambda plan, logits: False)
    assert _same_bits(*_pair(block, x[0]))
    assert _plan(block, x).proof.ran == ran | {(128, False)}
    assert [r.getMessage() for r in _logged(caplog, routed)] == [
        "Qwen MoE fused routed-expert decode engaged"
    ] * 3


def _quiet(launched, block, x):
    """The composed body's output for ``x``, its launches not recorded."""
    launched.recording = False
    ref = type(block).__call__._omlx_routed_decode_original(block, x)
    mx.eval(ref)
    launched.recording = True
    return ref


_TOKEN = ("router_gemv", "gate_up_topk", "down_combine_decode")


def _kinds(launches):
    """The launches' kernels, by what they run."""
    kinds = []
    for launch in launches:
        name = launch.name.removeprefix("omlx_qwen35_moe_")
        kinds.append(
            next(k for k in (*_TOKEN, "gate_up_decode", "router") if name.startswith(k))
        )
    return kinds


def test_a_block_s_launches_are_built_once_for_a_signature_and_replayed(
    launched, monkeypatch
):
    """MLX writes a kernel's source out for every launch that is built. A
    block's launches are one graph, built once for the token's shape and
    dtype and the router's width, and replayed; the blocks of a proof
    replay it over their own arrays."""
    compiled = _compiled(monkeypatch)
    decode, decoded = routed._decode, []
    monkeypatch.setattr(routed, "_decode", lambda *a: decoded.append(1) or decode(*a))
    block, twin = (_block(1024, 320, experts=128, seed=seed) for seed in (0, 1))
    xs = [
        (mx.random.normal((1, 1, 1024)) * (1 + s)).astype(mx.bfloat16) for s in range(3)
    ]
    for b in (block, twin):
        for x in xs:
            assert _same_bits(_quiet(launched, b, x), b(x))
    # The block's first call builds the graph; every other call is its replay,
    # looked up ahead of anything else the call asks.
    assert decoded == [1]
    # The stand-in's launches, then the graph's, built as MLX traces it.
    assert _kinds(launched.built) == [
        "gate_up_topk",
        "down_combine_decode",
        "router_gemv",
    ] + [*_TOKEN]
    assert [graph.calls for graph in compiled] == [6]
    plan = _plan(block, xs[0])
    assert _plan(twin, xs[0]).proof is plan.proof
    assert set(plan.proof.graphs) == {((1, 1, 1024), mx.bfloat16, 128)}
    # Another shape of the token is another graph, of the Metal proven.
    del launched.built[:]
    row = xs[0].reshape(1, 1024)
    assert _same_bits(_quiet(launched, block, row), block(row))
    assert _kinds(launched.built) == [*_TOKEN] and len(compiled) == 2
    assert plan.proof.ran == {(128, True)}


def test_a_graph_that_cannot_be_built_is_raised_and_not_kept(monkeypatch):
    """What fails as the call's launches are built, after their Metal is
    proven, is a defect of the kernel or of its caller: raised by the call,
    and nothing is kept for the signature."""
    block = _block(1024, 320, experts=128)
    x = _token_of(block)
    experts, built = routed._experts, []

    def refused(*args):
        built.append(1)
        if len(built) > 1:  # the stand-in's are built, the call's are not
            raise RuntimeError("no launch")
        return experts(*args)

    monkeypatch.setattr(routed, "_experts", refused)
    with pytest.raises(RuntimeError, match="no launch"):
        block(x)
    plan = _plan(block, x)
    assert plan.proof.ran == {(128, True)} and not plan.proof.failed
    assert not plan.proof.graphs
    monkeypatch.setattr(routed, "_experts", experts)
    assert _same_bits(*_pair(block, x))
    assert len(plan.proof.graphs) == 1


def test_routers_of_other_widths_are_graphs_of_their_own(monkeypatch):
    """Blocks of the same faces and shapes share a proof whatever their
    router's width, and the router's width is a template value of two
    launches: their graphs are kept apart."""
    compiled = _compiled(monkeypatch)
    x = mx.random.normal((1, 1, 512)).astype(mx.bfloat16)
    blocks = [_block(512, 256, top_k=8, experts=experts) for experts in (128, 256)]
    for block in blocks * 2:
        assert _same_bits(*_pair(block, x))
    proof = _plan(blocks[0], x).proof
    assert _plan(blocks[1], x).proof is proof
    assert set(proof.graphs) == {((1, 1, 512), mx.bfloat16, e) for e in (128, 256)}
    assert [graph.calls for graph in compiled] == [2, 2]


def test_a_launch_that_fails_drops_the_graphs_that_ran(monkeypatch, caplog):
    """A proof's launches stop together: a graph it kept is dropped with
    the failure of another of its launches, which composes from then on."""
    caplog.set_level(logging.WARNING, logger=routed.__name__)
    block = _block(512, 256, top_k=8, experts=128)
    x = mx.random.normal((1, 1, 512)).astype(mx.bfloat16)
    assert _same_bits(*_pair(block, x))
    proof = _plan(block, x).proof
    assert len(proof.graphs) == 1
    routing = routed._routing

    def broken(logits, top_k):
        """The routing launch, failing at its first evaluation."""
        return tuple(_does_not_compile(a) for a in routing(logits, top_k))

    monkeypatch.setattr(routed, "_topk_folds", lambda plan, logits: False)
    monkeypatch.setattr(routed, "_routing", broken)
    assert _same_bits(*_pair(block, x[0]))
    assert proof.failed and proof.ran == {(128, True)} and not proof.graphs
    assert len(_logged(caplog, routed, logging.WARNING)) == 1
    # The first signature composes too, and builds nothing.
    monkeypatch.setattr(mx, "compile", lambda *a: pytest.fail("built"))
    assert _same_bits(*_pair(block, x))
    assert not proof.graphs


def _selected(launch):
    """The experts that a launch selects, if it selects."""
    name = launch.name.removeprefix("omlx_qwen35_moe_")
    if name.startswith("gate_up_topk"):
        return launch.outputs[1]
    if name.startswith(("router_softmax_topk", "router_topk")):
        return launch.outputs[0]
    return None


@pytest.mark.parametrize(
    "experts,fold,quantized_shared",
    [
        (128, True, True),  # the routing inside the gate+up launch
        (128, False, True),  # the routing launch
        (EXPERTS, False, True),  # the block's own gate, a composed softmax
        (128, True, False),  # a shared expert outside the launches
    ],
)
def test_a_stand_in_reads_zeros_and_routes_inside_its_stacks(
    launched, monkeypatch, experts, fold, quantized_shared
):
    """A stand-in reads no array of the call: zeros for the token, the
    router's weight, each shared matrix and what the block computes outside
    the launches, and as many matrices of each stack as a row selects.
    Logits that rank the experts route the row to the first matrices, which
    the stacks have: no launch reads past them."""
    if not fold:
        monkeypatch.setattr(routed, "_topk_folds", lambda plan, logits: False)
    block = _block(1024, 320, experts=experts, quantized_shared=quantized_shared)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    plan = _plan(block, x)
    called = [x, *(a for group in plan.operands for a in group)]
    called += [a for _, a in tree_flatten(block.parameters())]
    assert _same_bits(_quiet(launched, block, x), block(x))
    # The stand-in's launches, then the call's: the same kernels.
    launches = launched.built
    half = len(launches) // 2
    stand_in = launches[:half]
    assert sorted(_kinds(stand_in)) == sorted(_kinds(launches[half:]))
    assert not any(a is b for launch in stand_in for a in launch.inputs for b in called)
    top_k = plan.launches.top_k
    stacks = [
        a
        for launch in stand_in
        for a in launch.inputs
        if a.ndim == 3 and a.shape[0] > 1
    ]
    assert len(stacks) == len(plan.operands[1]) + len(plan.operands[2])
    for a in stacks:
        assert a.shape[0] == top_k and mx.array_equal(a, mx.zeros_like(a)).item()
    (indices,) = (i for i in map(_selected, stand_in) if i is not None)
    assert sorted(indices.reshape(-1).tolist()) == list(range(top_k))
    # Every other input is zeros, or what a launch of the stand-in gave.
    given = {id(a) for launch in stand_in for a in launch.outputs}
    for launch in stand_in:
        for a in launch.inputs:
            if id(a) not in given and a.shape[-1] != experts:
                assert mx.array_equal(a, mx.zeros_like(a)).item()


def _sources(capfd):
    """The sources that MLX printed since they were last read, launch by
    launch."""
    out = capfd.readouterr().out
    opening = r"^Generated source code for `\w+`:\n```\n"
    return re.findall(opening + r"(.*?)\n```$", out, re.DOTALL | re.MULTILINE)


def test_a_stand_in_has_the_metal_of_the_launches_it_proves(
    launched, monkeypatch, capfd
):
    """MLX writes a launch's Metal from the kernel's source, the template's
    values and, input by input, the dtype, whether the input has fewer than
    8 elements and whether it has no axis. A block's stand-in is the
    smallest launch of which all of that is the call's: MLX generates the
    same sources for the two, byte for byte, kernel by kernel."""
    launched.verbose = True
    folds = routed._topk_folds
    calls = [
        (_block(1024, 320, experts=128), mx.bfloat16, folds),
        (_block(1024, 512, top_k=8, experts=256, dtype=mx.float16), mx.float16, folds),
        (_block(960, 320, bits=4, group_size=32, experts=128), mx.bfloat16, folds),
        (_block(1024, 320, experts=128, quantized_shared=False), mx.bfloat16, folds),
        (_block(1024, 320), mx.bfloat16, folds),
        (_block(1024, 320, quantized_shared=False), mx.bfloat16, folds),
        # The routing launch of a router that the gate+up launch could fold.
        (_block(1024, 384, bits=6, group_size=128, experts=128), mx.bfloat16, None),
    ]
    generated, names = set(), set()
    for block, dtype, fold in calls:
        kernels._PROOFS.clear()
        monkeypatch.setattr(
            routed, "_topk_folds", fold or (lambda launches, logits: False)
        )
        x = mx.random.normal((1, 1, block.gate["weight"].shape[-1])).astype(dtype)
        capfd.readouterr()
        _quiet(launched, block, x)
        capfd.readouterr()
        mx.eval(block(x))
        sources = _sources(capfd)
        half = len(sources) // 2
        assert half >= 2 and sorted(sources[:half]) == sorted(sources[half:])
        (proof,) = (
            p
            for (kernel, *_), p in kernels._PROOFS.items()
            if kernel == "routed_decode"
        )
        assert len(proof.ran) == 1 and not proof.failed
        generated.update(sources)
        names.update(
            launch.name.removeprefix("omlx_qwen35_moe_") for launch in launched.built
        )
        del launched.built[:]
    # Among them: the gemv router, the routing inside the gate+up launch and
    # on its own (one launch, or the softmax and one), a shared expert in and
    # outside the launches.
    kinds = {(name.split("_affine")[0], "_shared_" in name) for name in names}
    assert kinds == {
        ("router_gemv", False),
        ("router_topk", False),
        ("router_softmax_topk_row", False),
        ("gate_up_topk_bf16", True),
        ("gate_up_topk_bf16", False),
        ("gate_up_topk_scan", True),
        ("gate_up_decode", True),
        ("gate_up_decode", False),
        ("down_combine_decode", True),
        ("down_combine_decode", False),
    }
    assert len(generated) == 16


def _token_of(block, seed=0):
    mx.random.seed(100 + seed)
    return mx.random.normal((1, 1, block.gate["weight"].shape[-1])).astype(mx.bfloat16)


def test_a_proven_block_is_built_into_a_function_that_mlx_compiles(monkeypatch):
    """MLX's compile calls a block once for the shapes it is given, over an
    x that cannot be evaluated: a proven block needs nothing evaluated."""
    folded, composed = _block(1024, 320, experts=128), _block(1024, 320)
    x = _token_of(folded)
    folded(x), composed(x)
    evaluated = _evaluations(monkeypatch)
    graph = mx.compile(lambda x: (folded(x) * 2, composed(x) * 2))
    for seed in range(3):
        x = _token_of(folded, seed)
        y, z = graph(x)
        assert _same_bits(y, folded(x) * 2) and _same_bits(z, composed(x) * 2)
    assert not evaluated
    assert [len(_plan(b, x).proof.graphs) for b in (folded, composed)] == [1, 1]


def test_an_unproven_block_is_proven_in_a_function_that_mlx_compiles(
    monkeypatch, caplog
):
    """A block's first call may be the one MLX's compile makes, over an x
    that cannot be evaluated: the launches are proven on their stand-in,
    which reads no array of the call, and the graph is kept."""
    caplog.set_level(logging.INFO, logger=routed.__name__)
    block = _block(1024, 320, experts=128)
    x, later = _token_of(block), _token_of(block, 1)
    graph = mx.compile(lambda x: block(x) * 2)
    compiled = _compiled(monkeypatch)
    want = type(block).__call__._omlx_routed_decode_original(block, x) * 2
    assert _same_bits(graph(x), want)
    plan = _plan(block, x)
    assert plan.proof.ran == {(128, True)} and len(plan.proof.graphs) == 1
    assert [r.getMessage() for r in _logged(caplog, routed)] == [_ENGAGED]
    # Outside the function the kept graph is traced once, then replayed;
    # nothing is evaluated or proven again.
    evaluated = _evaluations(monkeypatch)
    for call in (later, x):
        assert _same_bits(
            block(call), type(block).__call__._omlx_routed_decode_original(block, call)
        )
        assert _same_bits(graph(call), block(call) * 2)
    assert len(compiled) == 1 and len(_logged(caplog, routed)) == 1
    assert not evaluated


_ENGAGED = "Qwen MoE fused routed-expert decode engaged"


def test_a_block_is_proven_and_kept_in_a_function_compiled_over_its_state(
    monkeypatch, caplog
):
    """``mx.compile(fn, inputs=state)`` hands a block its own tensors as
    arrays that cannot be evaluated, nor bound as a view of one expert: the
    launches are proven on their stand-in, built over the trace's stacks
    whole and kept, and the block keeps its tensors."""
    caplog.set_level(logging.INFO, logger=routed.__name__)
    block = _block(1024, 320, experts=128)
    x, later = _token_of(block), _token_of(block, 1)
    want = type(block).__call__._omlx_routed_decode_original(block, x)
    mx.eval(want)
    graph = mx.compile(lambda x: block(x) * 2, inputs=block.state)
    for _ in range(2):
        assert _same_bits(graph(x), want * 2)
    # The function left the block its own tensors.
    mx.eval(block.parameters())
    plan = _plan(block, x)
    assert plan.proof.ran == {(128, True)} and len(plan.proof.graphs) == 1
    assert all(a.shape[0] == 1 for a in plan.operands[1] + plan.operands[2])
    assert [r.getMessage() for r in _logged(caplog, routed)] == [_ENGAGED]
    for call in (later, x):
        assert _same_bits(*_pair(block, call))
    assert len(plan.proof.graphs) == 1


def test_a_face_that_does_not_compile_in_a_compiled_function_is_the_composed_body(
    monkeypatch, caplog
):
    """The stand-in fails as the launches would: the function is built over
    the block's composed body, in a plain trace and over the state."""
    caplog.set_level(logging.WARNING, logger=routed.__name__)
    dot = formats.dot

    def broken(layer, k, n):
        face = dot(layer, k, n)
        if face is None or k != 320:
            return face
        source = face.source + 'static_assert(false, "does not compile");\n'
        return replace(face, key=face.key + "_broken", source=source)

    monkeypatch.setattr(formats, "dot", broken)
    block = _block(1024, 320, experts=128)
    x = _token_of(block)
    want = type(block).__call__._omlx_routed_decode_original(block, x)
    mx.eval(want)
    for inputs in ([], block.state):
        graph = mx.compile(lambda x: block(x), inputs=inputs)
        assert _same_bits(graph(x), want)
    mx.eval(block.parameters())
    (warning,) = _logged(caplog, routed, logging.WARNING)
    assert "does not compile" in str(warning.exc_info[1])
    proof = kernels._PROOFS[next(k for k in kernels._PROOFS if k[0] == "routed_decode")]
    assert proof.failed and not proof.ran and not proof.graphs


class _Thread:
    """A thread that runs MLX as oMLX's engines do: one executor's worker."""

    def __init__(self, name):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self(lambda: mx.zeros((1,)).item())
        self.ident = self(threading.get_ident)

    def __call__(self, fn, *args):
        return self.pool.submit(fn, *args).result(timeout=60)

    def end(self):
        self.pool.shutdown()


def _called(block, x):
    y = block(x)
    mx.eval(y)
    return y


def test_a_block_s_graph_is_called_by_the_thread_that_built_it(monkeypatch):
    """MLX keeps what it traces a compiled function into thread by thread,
    and the function's end forgets it for its last caller alone. So each
    thread keeps the graphs it builds, calls no other's, and proves nothing
    that another thread proved."""
    block = _block(1024, 320, experts=128)
    x = _token_of(block)
    want = type(block).__call__._omlx_routed_decode_original(block, x)
    mx.eval(want)
    compiled = _compiled(monkeypatch)
    engines = [_Thread("mlx-engine-a"), _Thread("mlx-engine-b")]
    try:
        for run in (engines[0], engines[1], engines[0], engines[1]):
            assert _same_bits(run(_called, block, x), want)
        assert _same_bits(_called(block, x), want)
        idents = {engine.ident for engine in engines} | {threading.get_ident()}
        plan = _plan(block, x)
        counts = [engine(lambda: len(plan.proof.graphs)) for engine in engines]
    finally:
        for engine in engines:
            engine.end()
    assert len(compiled) == 3 and all(len(graph.threads) == 1 for graph in compiled)
    assert set.union(*(graph.threads for graph in compiled)) == idents
    assert counts == [1, 1] and plan.proof.ran == {(128, True)}


def test_a_kept_graph_is_not_replayed_for_a_token_of_another_dtype():
    """x's dtype is part of a graph's signature: a token in another dtype
    than the launches are compiled for is the composed body's, whatever
    graphs the shape has."""
    block = _block(1024, 512, top_k=8, experts=128)
    x = _token_of(block)
    assert _same_bits(*_pair(block, x))
    plan = _plan(block, x)
    kept = dict(plan.proof.graphs)
    half = x.astype(mx.float16)
    ref, out = _pair(block, half)
    assert out.dtype == ref.dtype and mx.array_equal(out, ref).item()
    assert plan.proof.graphs == kept and plan.proof.ran == {(128, True)}


@pytest.mark.parametrize("experts", [EXPERTS, 128])
def test_x_of_another_width_is_the_composed_body_s(experts):
    """A row of another width than the block's is not routed: the block's
    composed body refuses it, as before."""
    block = _block(1024, 320, experts=experts)
    assert _same_bits(*_pair(block, _token_of(block)))
    wide = mx.zeros((1, 1, 2048), mx.bfloat16)
    original = type(block).__call__._omlx_routed_decode_original
    with pytest.raises(ValueError) as composed:
        mx.eval(original(block, wide))
    with pytest.raises(ValueError) as routed_call:
        mx.eval(block(wide))
    assert str(routed_call.value) == str(composed.value)


def test_a_router_of_partial_simdgroups_keeps_the_composed_body():
    """The fused routing reads an expert count of whole simdgroups."""
    block = _block(1024, 320, experts=48)
    x = _token_of(block)
    assert _plan(block, x) is None
    assert _same_bits(*_pair(block, x))
    assert not any(proof.ran for proof in kernels._PROOFS.values())


@pytest.mark.parametrize("outside", ["shared", "shared_gate", "router"])
def test_a_kept_graph_takes_only_outside_arrays_of_the_token_s_dtype(outside):
    """A shared expert, its gate and a gate that run as they are give the
    graph an array each call: one in another dtype than the token's is not
    what the graph was built for, and the block composes."""
    if outside in ("shared", "shared_gate"):
        block = _block(1024, 320, experts=128, quantized_shared=False)
        name = {"shared": "shared_expert", "shared_gate": "shared_expert_gate"}[outside]
    else:
        block = _block(1024, 320)
        name = "gate"
    x = _token_of(block)
    assert _same_bits(*_pair(block, x))
    plan = _plan(block, x)
    assert len(plan.proof.graphs) == 1
    original = block[name]
    block[name] = _CastGate(original, mx.float16)
    assert _same_bits(*_pair(block, x))
    block[name] = original
    assert len(plan.proof.graphs) == 1 and plan.proof.ran and not plan.proof.failed


def test_apply_requires_the_fused_router(_patched_block):
    cls = _patched_block
    del cls._omlx_routed_decode
    cls._omlx_router_fused = False
    assert not routed.apply_qwen35_moe_routed_decode_patch()


def test_apply_loads_the_formats_or_leaves_the_block_unpatched(
    _patched_block, monkeypatch, caplog
):
    caplog.set_level(logging.ERROR, logger=routed.__name__)
    cls = _patched_block
    served = cls.__call__._omlx_routed_decode_original
    loads = []

    def unpatched():
        cls.__call__ = served
        del cls._omlx_routed_decode

    unpatched()
    monkeypatch.setattr(formats, "load", lambda: loads.append(1))
    assert routed.apply_qwen35_moe_routed_decode_patch()
    assert loads == [1]  # with the model, ahead of its first token

    def broken():
        raise ImportError("a built-in format does not import")

    unpatched()
    monkeypatch.setattr(formats, "load", broken)
    with pytest.raises(ImportError, match="built-in format"):
        routed.apply_qwen35_moe_routed_decode_patch()
    assert cls.__call__ is served and "_omlx_routed_decode" not in cls.__dict__
    # Said here: the engine's guard logs what it catches at debug level.
    (record,) = _logged(caplog, routed, logging.ERROR)
    message = "quant formats did not load; routed decode not applied"
    assert record.getMessage() == message
    assert type(record.exc_info[1]) is ImportError


def test_apply_is_idempotent(_patched_block):
    cls = _patched_block
    call = cls.__call__
    assert routed.apply_qwen35_moe_routed_decode_patch()
    assert cls.__call__ is call


class _Shaped(nn.Module):
    """A module whose output is reshaped or cut by ``reshape``."""

    def __init__(self, module, reshape):
        super().__init__()
        self.module = module
        self.reshape = reshape

    def __call__(self, x):
        return self.reshape(self.module(x))


@pytest.mark.parametrize(
    "name,reshape",
    [
        # A shared expert of half the width: the composed body refuses it.
        ("shared_expert", lambda y: y[..., : y.shape[-1] // 2]),
        # A gate value of no axis: the down launch reads gate[0].
        ("shared_expert_gate", lambda y: y.reshape(())),
        # A gate of two values: the down launch would read the first alone.
        ("shared_expert_gate", lambda y: mx.concatenate([y, y + 1], axis=-1)),
        # A router that scores fewer experts than the block has.
        ("gate", lambda y: y[..., :96]),
        # Logits of two rows for one token.
        ("gate", lambda y: mx.concatenate([y, y], axis=0)),
        # A shared output of the block's size in another shape.
        ("shared_expert", lambda y: y.reshape(2, 1, -1)),
        # A gate value on one more axis than the token's.
        ("shared_expert_gate", lambda y: y.reshape(1, 1, 1, 1)),
    ],
)
def test_what_the_block_computes_outside_the_launches_is_read_in_their_shapes(
    name, reshape
):
    """What a block computes outside its launches reaches them as it is: of
    another shape than the launches read, the block is its composed body's
    (which may refuse it), and no proof fails."""
    block = _block(1024, 320, experts=128, quantized_shared=name == "gate")
    x = _token_of(block)
    assert _same_bits(*_pair(block, x))
    plan = _plan(block, x)
    original = block[name]
    block[name] = _Shaped(original, reshape)
    composed = type(block).__call__._omlx_routed_decode_original
    try:
        want = composed(block, x)
        mx.eval(want)
    except ValueError as refused:
        with pytest.raises(ValueError, match=re.escape(str(refused))):
            mx.eval(block(x))
    else:
        got = block(x)
        assert got.dtype == want.dtype and mx.array_equal(got, want).item()
    finally:
        block[name] = original
    assert plan.proof.ran and not plan.proof.failed and len(plan.proof.graphs) == 1
    assert not any(proof.failed for proof in kernels._PROOFS.values())


def test_a_shared_expert_of_rows_the_launches_do_not_group_is_left_out(monkeypatch):
    """The launches take a shared expert in whole groups of a simdgroup's
    rows: of a width that is not, the shared expert runs as it is."""
    monkeypatch.setattr(routed, "_GATE_UP_ROWS", 3)  # groups of 6 rows
    block = _block(1024, 384, experts=128, shared_inter=320)
    x = _token_of(block)
    assert not _plan(block, x).launches.fold
    assert _same_bits(*_pair(block, x))


def test_a_shared_expert_whose_up_has_another_face_is_left_out():
    """The gate+up launch reads the shared expert's gate and up rows through
    one face: an up of other bits keeps the shared expert as it is."""
    block = _block(1024, 320, experts=128, quantized_shared=False)
    shared = block.shared_expert
    for name, bits, group in (
        ("gate_proj", 8, 128),
        ("up_proj", 4, 64),
        ("down_proj", 8, 64),
    ):
        setattr(
            shared,
            name,
            nn.QuantizedLinear.from_linear(getattr(shared, name), group, bits),
        )
    block.shared_expert_gate = nn.QuantizedLinear.from_linear(
        block.shared_expert_gate, 64, 8
    )
    mx.eval(block.parameters())
    x = _token_of(block)
    assert not _plan(block, x).launches.fold
    assert _same_bits(*_pair(block, x))


def test_stacks_of_other_expert_counts_are_not_planned():
    """Gate+up and down index one expert count."""
    block = _block(1024, 320, experts=128)
    down = block.switch_mlp.down_proj
    for name in ("weight", "scales", "biases"):
        down[name] = down[name][:-32]
    assert _plan(block, _token_of(block)) is None


def _rows_of(block, batch, length, seed=0):
    mx.random.seed(200 + seed)
    hidden = block.gate["weight"].shape[-1]
    return mx.random.normal((batch, length, hidden)).astype(mx.bfloat16)


def _one_by_one(block, x):
    """Each row of ``x`` through the block's one-token call: what a window's
    rows equal."""
    rows = x.reshape(-1, 1, 1, x.shape[-1])
    return mx.concatenate([block(row) for row in rows]).reshape(-1, x.shape[-1])


def test_a_window_s_launches_are_built_once_for_a_signature_and_replayed(
    launched, monkeypatch
):
    """A window's launches are one graph too, built once for the window's
    shape and dtype and the router's width and replayed, by every block of
    the window's proof; a window of other rows, and batched decode rows,
    are graphs of their own."""
    block, twin = (_block(512, 256, top_k=8, experts=128, seed=s) for s in (0, 1))
    launched.recording = False
    calls = [(b, _rows_of(b, 1, 3, seed)) for b in (block, twin) for seed in range(3)]
    wants = [_one_by_one(b, x) for b, x in calls]
    more = [_rows_of(block, 1, 8), _rows_of(block, 4, 1)]
    more_wants = [_one_by_one(block, x) for x in more]
    mx.eval(wants, more_wants)
    launched.recording = True
    compiled = _compiled(monkeypatch)
    for (b, x), want in zip(calls, wants, strict=True):
        assert _same_bits(routed.routed_verify_window(b, x), want)
    # The stand-in's launches, then the graph's: router, gate+up with the
    # routing, down; built once.
    assert len(launched.built) == 6
    assert [graph.calls for graph in compiled] == [6]
    plan = _plan(block, _rows_of(block, 1, 1))
    assert set(plan.window.graphs) == {((1, 3, 512), mx.bfloat16, 128)}
    # A verify window of 8 rows, and 4 requests' decode rows through the
    # block's call.
    window, rows = more
    assert _same_bits(routed.routed_verify_window(block, window), more_wants[0])
    assert _same_bits(block(rows).reshape(-1, 512), more_wants[1])
    assert set(plan.window.graphs) == {
        ((1, 3, 512), mx.bfloat16, 128),
        ((1, 8, 512), mx.bfloat16, 128),
        ((4, 1, 512), mx.bfloat16, 128),
    }
    assert plan.window.ran == {(3, 128, True), (8, 128, False), (4, 128, False)}
    assert len(compiled) == 3


def test_a_kept_window_is_not_replayed_for_rows_of_another_dtype():
    block = _block(512, 256, top_k=8, experts=128)
    x = _rows_of(block, 1, 3)
    assert routed.routed_verify_window(block, x) is not None
    plan = _plan(block, x[:, :1])
    kept = dict(plan.window.graphs)
    assert routed.routed_verify_window(block, x.astype(mx.float16)) is None
    assert plan.window.graphs == kept


def test_windows_of_no_rows_or_of_more_than_the_verify_kernels_take_are_composed():
    block = _block(512, 256, top_k=8, experts=128)
    assert routed.routed_verify_window(block, _rows_of(block, 1, 0)) is None
    # Axes other than a verify block's [B, L, hidden], of one row too.
    for x in (
        mx.zeros((), mx.bfloat16),
        _rows_of(block, 1, 1)[0],
        _rows_of(block, 1, 3)[None],
    ):
        assert routed.routed_verify_window(block, x) is None
    # One row of another dtype than the launches': one token's call declines.
    assert (
        routed.routed_verify_window(block, _rows_of(block, 1, 1).astype(mx.float16))
        is None
    )
    wide = _rows_of(block, 1, routed.WINDOW_MAX_ROWS + 1)
    assert routed.routed_verify_window(block, wide) is None
    # A window has the axes of a verify block's: [B, L, hidden].
    assert routed.routed_verify_window(block, _rows_of(block, 1, 3)[0]) is None
    assert (
        routed.routed_verify_window(block, wide[:, : routed.WINDOW_MAX_ROWS])
        is not None
    )
    plan = _plan(block, wide[:, :1])
    assert plan.window.ran == {(routed.WINDOW_MAX_ROWS, 128, False)}


def test_windows_of_routers_of_other_widths_are_graphs_of_their_own():
    blocks = [_block(512, 256, top_k=8, experts=e) for e in (128, 256)]
    for block in blocks * 2:
        x = _rows_of(block, 1, 3)
        assert _same_bits(routed.routed_verify_window(block, x), _one_by_one(block, x))
    window = _plan(blocks[0], _rows_of(blocks[0], 1, 1)).window
    assert _plan(blocks[1], _rows_of(blocks[1], 1, 1)).window is window
    assert set(window.graphs) == {((1, 3, 512), mx.bfloat16, e) for e in (128, 256)}
    assert window.ran == {(3, 128, True), (3, 256, True)}


def test_a_failed_one_token_launch_stops_the_windows_kept(monkeypatch, caplog):
    """A one-token launch that fails keeps the blocks' windows composed,
    whatever window graphs their threads kept."""
    caplog.set_level(logging.WARNING, logger=routed.__name__)
    block = _block(512, 256, top_k=8, experts=128)
    x = _rows_of(block, 1, 3)
    assert routed.routed_verify_window(block, x) is not None
    plan = _plan(block, x[:, :1])
    assert len(plan.window.graphs) == 1
    down = routed._down
    monkeypatch.setattr(routed, "_down", lambda *a: _does_not_compile(down(*a)))
    assert _same_bits(*_pair(block, x[:, :1]))
    assert plan.proof.failed and len(_logged(caplog, routed, logging.WARNING)) == 1
    assert routed.routed_verify_window(block, x) is None


def test_a_window_whose_stand_in_fails_drops_its_graphs(monkeypatch, caplog):
    """The windows of a proof stop together: the window graphs kept are
    dropped with a window that fails; one token still runs."""
    caplog.set_level(logging.WARNING, logger=routed.__name__)
    block = _block(512, 256, top_k=8, experts=128)
    x = _rows_of(block, 1, 3)
    assert routed.routed_verify_window(block, x) is not None
    window = _plan(block, x[:, :1]).window
    assert len(window.graphs) == 1
    down = routed._window_down
    monkeypatch.setattr(routed, "_window_down", lambda *a: _does_not_compile(down(*a)))
    assert routed.routed_verify_window(block, _rows_of(block, 1, 5)) is None
    assert window.failed and not window.graphs
    # A failed window builds nothing of a launch again.
    with monkeypatch.context() as patch:
        patch.setattr(routed, "_ranked", lambda *a: pytest.fail("built"))
        assert routed.routed_verify_window(block, x) is None
    warnings = _logged(caplog, routed, logging.WARNING)
    assert [r.getMessage() for r in warnings] == [
        "fused verify-window experts failed; verifier fallback"
    ]
    assert _same_bits(*_pair(block, x[:, :1]))


@pytest.mark.parametrize("rows,fold", [(2, True), (3, True), (4, False), (16, False)])
def test_a_window_s_stand_in_has_the_metal_of_its_launches_and_its_stacks(
    launched, capfd, rows, fold
):
    """A window's stand-in keeps the window's rows (a template value) over
    zeros, routes every row to the first matrices of stacks of top-k, and
    MLX generates the same sources for it as for the window's launches."""
    launched.verbose = True
    block = _block(512, 256, top_k=8, experts=128)
    x = _rows_of(block, 1, rows)
    plan = _plan(block, x[:, :1])
    called = [x, *(a for group in plan.operands for a in group)]
    capfd.readouterr()
    y = routed.routed_verify_window(block, x)
    mx.eval(y)
    sources = _sources(capfd)
    half = len(sources) // 2
    assert half >= 3 and sorted(sources[:half]) == sorted(sources[half:])
    stand_in = launched.built[: len(launched.built) // 2]
    assert not any(a is b for launch in stand_in for a in launch.inputs for b in called)
    stacks = [a for launch in stand_in for a in launch.inputs if a.ndim == 3]
    assert len(stacks) == len(plan.operands[1]) + len(plan.operands[2])
    assert all(a.shape[0] == 8 and not mx.any(a).item() for a in stacks)
    (indices,) = (i for i in map(_selected, stand_in) if i is not None)
    assert indices.shape == (rows, 8)
    assert all(sorted(row) == list(range(8)) for row in indices.tolist())
    assert plan.window.ran == {(rows, 128, fold)}


def test_a_window_is_proven_and_kept_in_functions_that_mlx_compiles(caplog):
    """A window's first call may be inside a function that MLX's compile
    traces, over the block's state too: proven on its stand-in and kept."""
    caplog.set_level(logging.INFO, logger=routed.__name__)
    block = _block(512, 256, top_k=8, experts=128)
    x = _rows_of(block, 1, 3)
    want = _one_by_one(block, x)
    mx.eval(want)
    window = mx.compile(lambda x: routed.routed_verify_window(block, x) * 2)
    assert _same_bits(window(x), want * 2)
    plan = _plan(block, x[:, :1])
    assert plan.window.ran == {(3, 128, True)} and len(plan.window.graphs) == 1
    stated = mx.compile(
        lambda x: routed.routed_verify_window(block, x), inputs=block.state
    )
    wide = _rows_of(block, 1, 5)
    wide_want = _one_by_one(block, wide)
    for _ in range(2):
        assert _same_bits(stated(wide), wide_want)
    mx.eval(block.parameters())
    plan = _plan(block, x[:, :1])
    assert plan.window.ran == {(3, 128, True), (5, 128, False)}
    assert len(plan.window.graphs) == 2
    engaged = [r.getMessage() for r in _logged(caplog, routed)]
    assert [m for m in engaged if "window" in m] == [
        "Qwen MoE fused verify-window experts engaged (3 rows)",
        "Qwen MoE fused verify-window experts engaged (5 rows)",
    ]


def test_a_window_s_graph_is_called_by_the_thread_that_built_it(monkeypatch):
    block = _block(512, 256, top_k=8, experts=128)
    x = _rows_of(block, 1, 3)
    want = _one_by_one(block, x)
    mx.eval(want)
    compiled = _compiled(monkeypatch)
    engines = [_Thread("mlx-engine-a"), _Thread("mlx-engine-b")]

    def window(x):
        y = routed.routed_verify_window(block, x)
        mx.eval(y)
        return y

    try:
        for run in (engines[0], engines[1], engines[0], engines[1]):
            assert _same_bits(run(window, x), want)
        assert _same_bits(window(x), want)
        idents = {engine.ident for engine in engines} | {threading.get_ident()}
        plan = _plan(block, x[:, :1])
        counts = [engine(lambda: len(plan.window.graphs)) for engine in engines]
    finally:
        for engine in engines:
            engine.end()
    assert len(compiled) == 3 and all(len(graph.threads) == 1 for graph in compiled)
    assert set.union(*(graph.threads for graph in compiled)) == idents
    assert counts == [1, 1] and plan.window.ran == {(3, 128, True)}

def test_a_window_of_one_row_is_one_token_s_call(monkeypatch):
    """A verify window of one row runs one token's launches: their graph and
    their proof, not the window's (cold-fused MEDIUM-1), so a window that
    failed leaves it fused."""
    block = _block(512, 256, top_k=8, experts=128)
    x = _rows_of(block, 1, 1)
    want = _one_by_one(block, x)
    mx.eval(want)
    plan = _plan(block, x)
    compiled = _compiled(monkeypatch)
    assert _same_bits(routed.routed_verify_window(block, x), want)
    # The one-token graph kept by the block's call, replayed.
    assert not compiled and len(plan.proof.graphs) == 1
    assert not plan.window.ran and not plan.window.graphs
    plan.window.fail()
    assert _same_bits(routed.routed_verify_window(block, x), want)


def test_a_window_graph_that_cannot_be_built_is_raised_and_not_kept(monkeypatch):
    """What fails as a window's launches are built, after their Metal is
    proven, is raised by the call, and nothing is kept for the window."""
    block = _block(512, 256, top_k=8, experts=128)
    x = _rows_of(block, 1, 3)
    experts, built = routed._experts, []

    def refused(*args):
        built.append(1)
        if len(built) > 1:  # the stand-in's are built, the call's are not
            raise RuntimeError("no launch")
        return experts(*args)

    monkeypatch.setattr(routed, "_experts", refused)
    with pytest.raises(RuntimeError, match="no launch"):
        routed.routed_verify_window(block, x)
    window = _plan(block, x[:, :1]).window
    assert window.ran == {(3, 128, True)} and not window.failed
    assert not window.graphs


def test_windows_the_launches_do_not_take_are_composed():
    """A window of another width than the block's, of a block whose gate
    runs as it is, or of a router of partial simdgroups is the verifier's:
    nothing is proven or kept."""
    block = _block(512, 256, top_k=8, experts=128)
    assert (
        routed.routed_verify_window(block, mx.zeros((1, 3, 1024), mx.bfloat16)) is None
    )
    gated = _block(1024, 320)
    plan = routed._build_plan(gated)
    assert plan.launches.router is None and plan.window is not None
    assert routed.routed_verify_window(gated, _rows_of(gated, 1, 3)) is None
    # 100 experts: the router gemv runs them, the routing kernels do not.
    partial_router = _block(1024, 320, experts=100)
    plan = routed._build_plan(partial_router)
    assert plan.launches.router is not None and plan.window is not None
    x = _rows_of(partial_router, 1, 3)
    assert routed.routed_verify_window(partial_router, x) is None
    assert not any(proof.ran or proof.graphs for proof in kernels._PROOFS.values())


class _ZerosError(Exception):
    """No exception oMLX or MLX raises."""


def test_what_fails_in_a_block_s_stand_in_zeros_is_raised_and_fails_no_proof(
    monkeypatch,
):
    """A block's and a window's stand-in inputs are made and evaluated ahead of
    the proof (``linear._zeros``): what fails there (an allocation, an error
    an earlier evaluation left pending on the GPU) is the call's, raised, and
    neither proof fails or proves anything."""
    block = _block(512, 256, top_k=8, experts=128)
    x, rows = _rows_of(block, 1, 1), _rows_of(block, 1, 3)
    error = _ZerosError("no zeros")
    with monkeypatch.context() as patch:
        patch.setattr(linear, "_zeros", lambda *specs: (_ for _ in ()).throw(error))
        for call in (
            lambda: block(x),
            lambda: routed.routed_verify_window(block, rows),
        ):
            with pytest.raises(_ZerosError):
                call()
    plan = _plan(block, x)
    for proof in (plan.proof, plan.window):
        assert not proof.failed and not proof.ran and not proof.graphs
    assert _same_bits(
        routed.routed_verify_window(block, rows), _one_by_one(block, rows)
    )
    assert plan.proof.ran and plan.window.ran


def test_a_block_s_stand_in_is_made_and_waited_for_on_the_gpu(monkeypatch):
    """Under a caller's CPU stream, the stand-in's logits and zeros are made
    on the GPU's stream and evaluated, and the stream is synchronized, before
    the launch is evaluated: one buffer of zeros for all of them."""
    block = _block(1024, 320, experts=128)
    x = _token_of(block)
    want = type(block).__call__._omlx_routed_decode_original(block, x)
    mx.eval(want)
    _plan(block, x)  # evaluates the experts' views
    evaluate, synchronize, zeros, arange = (
        mx.eval,
        mx.synchronize,
        linear._zeros,
        mx.arange,
    )
    events, streams = [], []

    def evaluated(*arrays):
        events.append("eval")
        evaluate(*arrays)

    def synchronized(*stream):
        events.append(("synchronize", *stream))
        synchronize(*stream)

    def made(*specs):
        events.append("zeros")
        return zeros(*specs)

    def ranked(*args, **named):
        streams.append(named.get("stream"))
        return arange(*args, **named)

    with monkeypatch.context() as patch:
        patch.setattr(mx, "eval", evaluated)
        patch.setattr(mx, "synchronize", synchronized)
        patch.setattr(mx, "arange", ranked)
        patch.setattr(linear, "_zeros", made)
        with mx.stream(mx.cpu):
            y = block(x)
    # The logits, the zeros (their evaluation, the synchronize), the launch.
    assert events == ["eval", "zeros", "eval", ("synchronize", mx.gpu), "eval"]
    assert streams == [mx.gpu]
    assert _same_bits(want, y)


def test_a_router_that_scores_other_experts_than_the_stacks_is_not_planned():
    """The router, the stacks and the block name one expert count: routing to
    an expert past the stacks would read past their buffers."""
    block = _block(1024, 320, experts=128)
    gate = nn.Linear(1024, 256, bias=False)
    gate.set_dtype(mx.bfloat16)
    block.gate = gate
    assert routed._build_plan(block) is None
    # Even where the block says so.
    block.num_experts = 256
    assert routed._build_plan(block) is None
    block = _block(1024, 320, experts=128)
    block.num_experts = 256
    assert routed._build_plan(block) is None
    assert routed._build_plan(_block(1024, 320, experts=128)) is not None


def test_a_kept_window_follows_the_block_s_replaced_weights():
    """A window's plan is keyed two levels down: weights replaced after a
    window graph was kept are the ones the next window reads."""
    block = _block(512, 256, top_k=8, experts=128)
    rows = _rows_of(block, 1, 3)
    assert _same_bits(
        routed.routed_verify_window(block, rows), _one_by_one(block, rows)
    )
    down = block.switch_mlp.down_proj
    down["scales"] = down["scales"] * 2
    # Against the composed body row by row, which keeps no plan.
    composed = type(block).__call__._omlx_routed_decode_original
    want = mx.concatenate([composed(block, row[None, None]) for row in rows[0]])
    got = routed.routed_verify_window(block, rows)
    assert _same_bits(got, want.reshape(got.shape))


def test_x_of_no_axis_is_refused_as_the_composed_body_refuses_it():
    block = _block(1024, 320, experts=128)
    x = mx.array(1.0, dtype=mx.bfloat16)
    composed = type(block).__call__._omlx_routed_decode_original
    with pytest.raises(Exception) as refused:
        mx.eval(composed(block, x))
    with pytest.raises(type(refused.value)):
        mx.eval(block(x))


def test_a_router_launcher_holds_no_array():
    """A kept graph closes over no array: the launcher takes the weight as
    an argument."""
    from omlx.patches.qwen35_moe_router import router_gemv

    weight = (mx.random.normal((128, 1024)) * 0.02).astype(mx.bfloat16)
    launch = router_gemv(weight)
    held = [cell.cell_contents for cell in launch.__closure__ or ()]
    assert held and not any(isinstance(value, mx.array) for value in held)
