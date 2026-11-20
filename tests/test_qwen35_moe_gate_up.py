# SPDX-License-Identifier: Apache-2.0
"""Tests for the supported MoE gate+up fusion patch (issue #2238)."""

from __future__ import annotations

import gc
import logging

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.switch_layers import SwitchGLU

import omlx.patches.qwen35_moe_gate_up as patch_mod
from omlx import formats
from omlx.formats import linear as format_linear
from omlx.formats import testing
from omlx.formats.linear import FormatSwitchLinear
from omlx.patches.qwen35_moe_gate_up import apply_qwen35_moe_gate_up_fusion
from tests.test_m5_gather_qmm_nax import qwen_regroup as qwen_regroup
from tests.test_moe_verify_gather import _proofs as _gather_proofs
from tests.test_qwen35_moe_routed_decode import format_checkpoint as format_checkpoint
from tests.test_qwen35_moe_routed_decode import load_toy_format

pytestmark = pytest.mark.usefixtures("qwen_regroup")

E, TOPK, HIDDEN, INTER = 8, 2, 64, 32


class _FakeQwenModel:
    # Module path carries the family token used by the gate.
    pass


_FakeQwenModel.__module__ = "mlx_lm.models.qwen3_5_moe"


class _FakeQwen4Model:
    pass


_FakeQwen4Model.__module__ = "mlx_vlm.models.qwen4_exp.qwen4_exp"


class _FakeOtherModel:
    pass


_FakeOtherModel.__module__ = "mlx_lm.models.deepseek_v3"


class _FakeHyV3Model:
    pass


_FakeHyV3Model.__module__ = "mlx_lm.models.hy_v3"


def _make_model(
    quantize=True,
    model_cls=_FakeQwenModel,
    n_blocks=2,
    group_size=32,
    bits=4,
):
    mx.random.seed(7)
    blocks = []
    for _ in range(n_blocks):
        glu = SwitchGLU(HIDDEN, INTER, E)
        if quantize:
            glu.gate_proj = glu.gate_proj.to_quantized(group_size, bits)
            glu.up_proj = glu.up_proj.to_quantized(group_size, bits)
            glu.down_proj = glu.down_proj.to_quantized(32, 4)
        blocks.append(glu)
    model = model_cls()
    model.blocks = blocks
    model.named_modules = lambda: [(f"blocks.{i}", b) for i, b in enumerate(blocks)]
    return model


def _forward_all(model, x, indices):
    return [blk(x, indices) for blk in model.blocks]


@pytest.mark.parametrize("quantize", [True, False])
def test_fused_output_bit_exact(quantize):
    model = _make_model(quantize=quantize)
    x = (mx.random.normal(shape=(1, 1, HIDDEN)) * 0.5).astype(mx.bfloat16)
    idx_decode = mx.random.randint(0, E, shape=(1, 1, TOPK))
    # 40 tokens x top-2 = 80 indices >= 64 exercises the sorted branch.
    x_sorted = (mx.random.normal(shape=(1, 40, HIDDEN)) * 0.5).astype(mx.bfloat16)
    idx_sorted = mx.random.randint(0, E, shape=(1, 40, TOPK))

    ref_decode = _forward_all(model, x, idx_decode)
    ref_prefill = _forward_all(model, x_sorted, idx_sorted)
    mx.eval(ref_decode, ref_prefill)

    fused = apply_qwen35_moe_gate_up_fusion(model)
    assert fused == 2
    for blk in model.blocks:
        assert hasattr(blk, "gate_up_proj")
        assert not hasattr(blk, "gate_proj")
        assert not hasattr(blk, "up_proj")

    out_decode = _forward_all(model, x, idx_decode)
    out_prefill = _forward_all(model, x_sorted, idx_sorted)
    mx.eval(out_decode, out_prefill)

    for ref, out in zip(ref_decode, out_decode):
        assert mx.array_equal(ref, out).item()
    for ref, out in zip(ref_prefill, out_prefill):
        assert mx.array_equal(ref, out).item()


def test_laguna_family_fused_bit_exact():
    """The vendored laguna model fuses its nvfp4 SwitchGLU experts."""
    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()

    from mlx_lm.models import laguna

    args = laguna.ModelArgs(
        model_type="laguna",
        vocab_size=256,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        layer_types=["full_attention", "full_attention"],
        mlp_only_layers=[],
        num_experts=E,
        num_experts_per_tok=TOPK,
        moe_intermediate_size=INTER,
        shared_expert_intermediate_size=INTER,
    )
    mx.random.seed(11)
    model = laguna.Model(args)
    for layer in model.model.layers:
        sw = layer.mlp.switch_mlp
        sw.gate_proj = sw.gate_proj.to_quantized(16, 4, mode="nvfp4")
        sw.up_proj = sw.up_proj.to_quantized(16, 4, mode="nvfp4")
        sw.down_proj = sw.down_proj.to_quantized(16, 4, mode="nvfp4")

    x = mx.array([[3, 1, 4, 1, 5]])
    ref = model(x)
    mx.eval(ref)

    fused = apply_qwen35_moe_gate_up_fusion(model)
    assert fused == 2
    for layer in model.model.layers:
        assert hasattr(layer.mlp.switch_mlp, "gate_up_proj")
        assert layer.mlp.switch_mlp.gate_up_proj.mode == "nvfp4"

    out = model(x)
    mx.eval(out)
    assert mx.array_equal(ref, out).item()


@pytest.mark.parametrize("bits", [5, 6, 8])
def test_hy_v3_family_fused_bit_exact(bits):
    """HyV3 uses stock SwitchGLU and receives the same bit-exact fusion."""
    model = _make_model(model_cls=_FakeHyV3Model, group_size=64, bits=bits)
    x = (mx.random.normal(shape=(1, 1, HIDDEN)) * 0.5).astype(mx.bfloat16)
    indices = mx.random.randint(0, E, shape=(1, 1, TOPK))

    ref = _forward_all(model, x, indices)
    mx.eval(ref)

    assert apply_qwen35_moe_gate_up_fusion(model) == 2
    out = _forward_all(model, x, indices)
    mx.eval(out)

    for expected, actual in zip(ref, out):
        assert mx.array_equal(expected, actual).item()


def test_qwen4_exp_family_is_eligible_for_gate_up_fusion():
    model = _make_model(model_cls=_FakeQwen4Model, n_blocks=1)

    assert apply_qwen35_moe_gate_up_fusion(model) == 1
    assert hasattr(model.blocks[0], "gate_up_proj")


def test_fusion_loads_the_formats_or_leaves_the_model_unfused(monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger=patch_mod.__name__)
    # Built first: quantizing a layer looks its mode's format up, which loads.
    other = _make_model(model_cls=_FakeOtherModel)
    model, unfused = _make_model(n_blocks=1), _make_model(n_blocks=1)
    loads = []
    monkeypatch.setattr(formats, "load", lambda: loads.append(1))
    assert apply_qwen35_moe_gate_up_fusion(other) == 0
    assert not loads  # nothing to fuse
    assert apply_qwen35_moe_gate_up_fusion(model) == 1
    assert loads == [1]  # with the model, ahead of its first verify block
    assert apply_qwen35_moe_gate_up_fusion(model) == 0
    assert loads == [1]  # a supported model with nothing left to fuse

    def broken():
        raise ImportError("a built-in format does not import")

    monkeypatch.setattr(formats, "load", broken)
    model = unfused
    with pytest.raises(ImportError, match="built-in format"):
        apply_qwen35_moe_gate_up_fusion(model)
    glu = model.blocks[0]
    assert hasattr(glu, "gate_proj") and not hasattr(glu, "gate_up_proj")
    # Said here: the engine's guard logs what it catches at debug level.
    (record,) = [r for r in caplog.records if r.name == patch_mod.__name__]
    message = "quant formats did not load; MoE gate+up fusion not applied"
    assert record.levelno == logging.ERROR and record.getMessage() == message
    assert type(record.exc_info[1]) is ImportError


def test_unsupported_family_skipped():
    model = _make_model(model_cls=_FakeOtherModel)
    assert apply_qwen35_moe_gate_up_fusion(model) == 0
    assert hasattr(model.blocks[0], "gate_proj")


def test_per_layer_pool_drain(monkeypatch):
    calls = []
    monkeypatch.setattr(patch_mod, "_sync_and_clear_cache", lambda: calls.append(1))

    skipped = _make_model(model_cls=_FakeOtherModel)
    assert apply_qwen35_moe_gate_up_fusion(skipped) == 0
    assert not calls

    model = _make_model(n_blocks=3)
    assert apply_qwen35_moe_gate_up_fusion(model) == 3
    assert len(calls) == 3


def test_idempotent():
    model = _make_model()
    assert apply_qwen35_moe_gate_up_fusion(model) == 2
    assert apply_qwen35_moe_gate_up_fusion(model) == 0


def test_mismatched_quant_params_skipped():
    model = _make_model(quantize=False, n_blocks=1)
    glu = model.blocks[0]
    glu.gate_proj = glu.gate_proj.to_quantized(32, 4)
    glu.up_proj = glu.up_proj.to_quantized(32, 8)
    glu.down_proj = glu.down_proj.to_quantized(32, 4)
    assert apply_qwen35_moe_gate_up_fusion(model) == 0
    assert hasattr(glu, "gate_proj")


@pytest.mark.parametrize("length", [1, 3, 64])
def test_vlm_fused_experts_preserve_decode_verify_and_prefill(length):
    from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

    mx.random.seed(7)
    glu = VLMSwitchGLU(HIDDEN, INTER, E)
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(glu, name, getattr(glu, name).to_quantized(32, 4))
    glu.eval()
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("experts", glu)]
    x = (mx.random.normal((2, length, HIDDEN)) * 0.5).astype(mx.bfloat16)
    idx = mx.random.randint(0, E, shape=(2, length, TOPK))
    weights = mx.full((2, length, TOPK), 0.5)
    shared = mx.zeros_like(x)
    ref = glu(x, idx, weights, shared)
    mx.eval(ref)

    assert apply_qwen35_moe_gate_up_fusion(model) == 1
    out = glu(x, idx, weights, shared)
    mx.eval(out)
    assert mx.array_equal(ref, out).item()


def test_weighted_sum_route_accepts_fused_layout():
    from omlx.patches.qwen35_moe_weighted_sum import _should_route

    model = _make_model(n_blocks=1)
    apply_qwen35_moe_gate_up_fusion(model)

    class _Block:
        top_k = 8
        sharding_group = None
        switch_mlp = model.blocks[0]

    if not mx.metal.is_available():
        pytest.skip("Metal required for _should_route")
    x = mx.zeros((1, 2048, HIDDEN), dtype=mx.bfloat16)
    assert _should_route(_Block(), x, target_verify=False, min_tokens=1024)


def test_vlm_fused_projection_views_cross_execution_threads():
    from concurrent.futures import ThreadPoolExecutor

    from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

    def load():
        with mx.stream(mx.new_thread_local_stream(mx.gpu)):
            glu = VLMSwitchGLU(HIDDEN, INTER, E)
            glu.gate_proj = glu.gate_proj.to_quantized(32, 4)
            glu.up_proj = glu.up_proj.to_quantized(32, 4)
            mx.eval(glu.parameters())
            patch_mod._fuse_one(glu)
            return glu

    def verify(glu):
        with mx.stream(mx.new_thread_local_stream(mx.gpu)):
            x = mx.ones((1, 1, 1, HIDDEN))
            indices = mx.zeros((1, TOPK), dtype=mx.uint32)
            outputs = [glu.gate_proj(x, indices), glu.up_proj(x, indices)]
            mx.eval(outputs)
            return all(bool(mx.all(mx.isfinite(value)).item()) for value in outputs)

    with ThreadPoolExecutor(max_workers=1) as loader:
        glu = loader.submit(load).result()
    with ThreadPoolExecutor(max_workers=1) as decoder:
        assert decoder.submit(verify, glu).result()


@pytest.mark.parametrize("bits,dtype", [(4, mx.bfloat16), (5, mx.float16)])
@pytest.mark.parametrize("batch,length", [(1, 3), (2, 6), (16, 2)])
def test_vlm_fused_short_block_matches_native_references_without_copying_views(
    monkeypatch, bits, dtype, batch, length
):
    from mlx_vlm.models import fast_ops
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward
    from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

    mx.random.seed(19)
    glu = VLMSwitchGLU(512, 64, 8)
    glu.set_dtype(dtype)
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(glu, name, getattr(glu, name).to_quantized(32, bits))
    glu.eval()
    verifier = Qwen3_5BatchInvariantForward()
    x = mx.random.normal((batch, length, 512)).astype(dtype)
    indices = mx.random.randint(0, 8, (batch, length, 6))
    # Use real one-token decode: mlx-vlm's short-block helpers still copy
    # the pre-0.32.4 QMV bias-sum arithmetic.
    expected_head = mx.concatenate(
        [glu(x[:, i : i + 1], indices[:, i : i + 1]) for i in range(length)],
        axis=1,
    )
    expected_verify = verifier._switch_glu(glu, x, indices)
    mx.eval(expected_head, expected_verify)

    model = _FakeQwen4Model()
    model.named_modules = lambda: [("experts", glu)]
    assert apply_qwen35_moe_gate_up_fusion(model) == 1

    def reject_view_kernel(*args):
        pytest.fail("Fused verification must not materialize gate/up views")

    monkeypatch.setattr(fast_ops, "exact_affine_switch_gate_up", reject_view_kernel)
    head = glu(x, indices)
    verified = verifier._switch_glu(glu, x, indices)
    mx.eval(head, verified)
    assert mx.array_equal(head, expected_head).item()
    assert mx.array_equal(verified, expected_verify).item()


@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_expert_ordered_verify_gather_matches_gather_qmm(bits):
    from mlx_vlm.models.switch_layers import SwitchLinear as VLMSwitchLinear

    from omlx.patches import moe_verify_gather

    mx.random.seed(bits)
    experts, top_k = 16, 10
    # K = 1024 takes the qmv_fast traversal, K = 320 the qmv one with a tail.
    for k, n in ((1024, 64), (320, 1024)):
        linear = VLMSwitchLinear(k, n, experts, bias=False)
        linear.set_dtype(mx.bfloat16)
        linear = linear.to_quantized(group_size=32, bits=bits)
        projection = moe_verify_gather.projection(linear, k)
        assert projection.face.exact_for(mx.bfloat16)
        for rows in (2, 8):
            indices = mx.stack(
                [mx.random.permutation(experts)[:top_k] for _ in range(rows)]
            ).astype(mx.uint32)
            x = mx.random.normal((rows, k)).astype(mx.bfloat16)
            expected = mx.gather_qmm(
                mx.expand_dims(x, (-2, -3)),
                linear["weight"],
                linear["scales"],
                linear["biases"],
                rhs_indices=indices,
                transpose=True,
                group_size=32,
                bits=bits,
            ).reshape(rows * top_k, n)
            got = moe_verify_gather.gather_qmv(
                projection, x, indices.reshape(-1), top_k
            )
            assert mx.array_equal(got, expected).item()


def _vlm_decode_switch(bits, seed):
    from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

    mx.random.seed(seed)
    glu = VLMSwitchGLU(2560, 640, 16)
    glu.set_dtype(mx.bfloat16)
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(glu, name, getattr(glu, name).to_quantized(64, bits))
    glu.eval()
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("experts", glu)]
    assert apply_qwen35_moe_gate_up_fusion(model) == 1
    return glu


def _assert_decode_plan_matches_per_call(monkeypatch, glu, seed):
    mx.random.seed(seed)
    for batch in (1, 3):
        x = (mx.random.normal((batch, 1, 2560)) * 0.5).astype(mx.bfloat16)
        idx = mx.random.randint(0, 16, shape=(batch, 1, 10)).astype(mx.uint32)
        with monkeypatch.context() as per_call:
            per_call.setattr(patch_mod, "cached_per_module", lambda *a: None)
            ref = glu(x, idx)
            mx.eval(ref)
        out = glu(x, idx)
        mx.eval(out)
        assert out.shape == ref.shape and out.dtype == ref.dtype == mx.bfloat16
        assert mx.array_equal(out.view(mx.uint16), ref.view(mx.uint16)).item()


@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("seed", [1, 2])
def test_vlm_decode_plan_is_bit_identical_to_per_call_path(monkeypatch, bits, seed):
    glu = _vlm_decode_switch(bits, seed)
    _assert_decode_plan_matches_per_call(monkeypatch, glu, seed)


def test_vlm_decode_plan_follows_replaced_experts(monkeypatch):
    from mlx_vlm.models.switch_layers import SwitchLinear as VLMSwitchLinear

    glu = _vlm_decode_switch(5, 3)
    _assert_decode_plan_matches_per_call(monkeypatch, glu, 3)
    down = VLMSwitchLinear(640, 2560, 16, bias=False)
    down.set_dtype(mx.bfloat16)
    glu.down_proj = down.to_quantized(64, 4)
    _assert_decode_plan_matches_per_call(monkeypatch, glu, 4)
    gate_up = glu.gate_up_proj
    gate_up.weight = mx.random.randint(
        0, 2**32 - 1, gate_up.weight.shape, dtype=mx.uint32
    )
    gate_up.scales = (gate_up.scales * 0.5).astype(gate_up.scales.dtype)
    _assert_decode_plan_matches_per_call(monkeypatch, glu, 5)


# A switch block whose experts load as a quant format's layers (through
# omlx.patches.format_load, as a checkpoint of a plugin format loads).
_FORMAT_CHECKPOINT = {"group_size": 1, "bits": 8, "mode": "toy"}
_FORMAT_STACKS = (
    ("gate_proj", INTER, HIDDEN),
    ("up_proj", INTER, HIDDEN),
    ("down_proj", HIDDEN, INTER),
)


def _format_glu(experts=E, cls=None):
    """mlx-vlm's SwitchGLU (or ``cls``) loaded as the toy format's stacks:
    one byte a weight and a float32 scale a row (tests.test_formats_load)."""
    from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

    from tests.test_formats_kernels import _matrix

    glu = (cls or VLMSwitchGLU)(HIDDEN, INTER, experts)
    nn.quantize(glu, **_FORMAT_CHECKPOINT)
    for i, (name, n, k) in enumerate(_FORMAT_STACKS):
        (codes, scales), _, _ = _matrix(experts * n, k, seed=20 + i)
        weights = [("weight", codes.reshape(experts, n, k))]
        # Scales of their own for each projection (powers of two: exact).
        weights.append(("scales", scales.reshape(experts, n) * 2.0**i))
        glu[name].load_weights(weights, strict=True)
    glu.eval()
    mx.eval(glu.parameters())
    return glu


@pytest.fixture
def loaded_format(monkeypatch, format_checkpoint):
    """The toy format, registered and loadable, with the face of the kernel
    tests."""
    from tests.test_formats_kernels import _toy

    return load_toy_format(monkeypatch, _toy("gate_up_fusion"))


@pytest.mark.parametrize("length", [1, 3, 64])
def test_a_block_of_format_stacks_fuses_its_gate_and_up(loaded_format, length):
    """The [gate; up] concatenation of each of the format's tensors along
    the rows is one stack of the format: decode, verify blocks (the verify
    gather) and prefill keep their bits."""
    glu = _format_glu()
    gate, up = glu.gate_proj, glu.up_proj
    assert type(gate) is FormatSwitchLinear
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("experts", glu)]
    mx.random.seed(length)
    x = mx.random.randint(-8, 9, (2, length, HIDDEN)).astype(mx.float32)
    idx = mx.random.randint(0, E, shape=(2, length, TOPK)).astype(mx.uint32)
    weights = mx.full((2, length, TOPK), 0.5)
    ref = glu(x, idx, weights, mx.zeros_like(x))
    mx.eval(ref)

    assert apply_qwen35_moe_gate_up_fusion(model) == 1
    fused = glu.gate_up_proj
    assert type(fused) is FormatSwitchLinear
    assert (fused.mode, fused.bits, fused.group_size) == ("toy", 8, 1)
    assert fused.input_dims == HIDDEN and fused.output_dims == 2 * INTER
    for name in ("weight", "scales"):
        assert mx.array_equal(
            fused[name], mx.concatenate([gate[name], up[name]], axis=1)
        ).item()
    out = glu(x, idx, weights, mx.zeros_like(x))
    mx.eval(out)
    assert mx.array_equal(ref, out).item()
    # A verify block of the fused stack takes the verify gather.
    proofs = [(bool(p.ran), p.failed) for p in _gather_proofs()]
    assert proofs == ([(True, False)] if length == 3 else [])


def test_format_stacks_that_are_not_one_layout_row_by_row_stay_unfused(loaded_format):
    """Fused only when each tensor of the two stacks has the same dtype and
    shape, the stacks' rows as its second axis, at the same bits and group
    size."""
    from tests.test_formats_kernels import _matrix

    def stack(n=INTER, bits=8, scales_shape=None, scales_dtype=mx.float32):
        (codes, scales), _, _ = _matrix(E * n, HIDDEN)
        scales = scales.reshape(E, n) if scales_shape is None else mx.ones(scales_shape)
        tensors = {
            "weight": codes.reshape(E, n, HIDDEN),
            "scales": scales.astype(scales_dtype),
        }
        return FormatSwitchLinear("toy", bits, 1, HIDDEN, tensors)

    for gate, up in (
        (stack(bits=7), stack()),  # other bits
        (stack(), stack(n=2 * INTER)),  # other rows
        (stack(scales_dtype=mx.float16), stack()),
        (stack(), stack(scales_shape=(E, INTER, 2))),  # rows of other shapes
        (stack(scales_shape=(E, 1)), stack(scales_shape=(E, 1))),  # not a row's
        (stack(scales_shape=(E,)), stack(scales_shape=(E,))),
    ):
        glu = _format_glu()
        glu.gate_proj, glu.up_proj = gate, up
        model = _FakeQwen4Model()
        model.named_modules = lambda glu=glu: [("experts", glu)]
        assert apply_qwen35_moe_gate_up_fusion(model) == 0
        assert glu.gate_proj is gate and "gate_up_proj" not in glu
    glu = _format_glu()
    glu.gate_proj, glu.up_proj = stack(), stack()
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("experts", glu)]
    assert apply_qwen35_moe_gate_up_fusion(model) == 1


@pytest.mark.parametrize("lm", [False, True])
def test_a_fused_format_stack_keeps_no_plan_of_its_parts(loaded_format, lm):
    """A layer's plan keeps the arrays it was built over: the fused stack
    (and mlx-vlm's gate and up views) start without one, so no plan holds
    the stacks of before the fusion."""
    glu = _format_glu(cls=SwitchGLU if lm else None)
    x = mx.random.randint(-8, 9, (1, 1, HIDDEN)).astype(mx.float32)
    idx = mx.random.randint(0, E, shape=(1, 1, TOPK)).astype(mx.uint32)
    mx.eval(glu(x, idx))
    assert all(format_linear._KEPT in vars(glu[n]) for n in ("gate_proj", "up_proj"))
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("experts", glu)]
    assert apply_qwen35_moe_gate_up_fusion(model) == 1
    layers = ("gate_up_proj",) if lm else ("gate_up_proj", "gate_proj", "up_proj")
    for name in layers:
        assert format_linear._KEPT not in vars(glu[name]), name


def _assert_kept_views_share_the_fused_buffer():
    """Fuse a VLM format stack (with a bias, which takes the axis=-1
    split) and assert its kept gate/up views share the fused buffers."""
    gc.collect()
    mx.clear_cache()
    glu = _format_glu()
    for proj in (glu.gate_proj, glu.up_proj):
        proj["bias"] = mx.zeros((E, INTER), mx.float32)
    mx.eval(glu.parameters())
    mx.synchronize()
    gc.collect()
    mx.clear_cache()
    gate, up = glu.gate_proj, glu.up_proj
    names = sorted(gate._tensors) + ["bias"]
    before = mx.get_active_memory()
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("experts", glu)]
    assert apply_qwen35_moe_gate_up_fusion(model) == 1
    mx.synchronize()
    fused = glu.gate_up_proj
    # A copy adds the routed bytes; a collected cycle can only free some.
    assert mx.get_active_memory() <= before
    for name in names:
        assert fused[name].nbytes == gate[name].nbytes + up[name].nbytes
        assert mx.get_array_buffer_size(gate[name]) == mx.get_array_buffer_size(
            fused[name]
        )
        assert mx.get_array_buffer_size(up[name]) == mx.get_array_buffer_size(
            fused[name]
        )
        assert mx.get_array_buffer_size(
            fused[name], gate[name], up[name]
        ) == mx.get_array_buffer_size(fused[name])


def test_vlm_kept_gate_up_views_share_the_fused_buffer(loaded_format):
    """mlx-vlm keeps gate/up for the verifier: ``mx.split`` returns views
    of the fused stack (MLX 0.32.4), so the routed gate/up bytes stay
    resident once: each view reports the fused buffer, the three report
    one buffer together, and active memory does not grow."""
    _assert_kept_views_share_the_fused_buffer()


def test_vlm_kept_gate_up_views_ignore_warm_buffer_cache(loaded_format):
    """The reading holds on a warm allocator: a cached three-page buffer
    is not reused as the fused stack's (the cleared cache and the ``<=``
    bound keep the active-memory check from measuring the cache)."""
    warm = mx.zeros((49152,), dtype=mx.uint8)
    mx.eval(warm)
    del warm
    mx.synchronize()
    _assert_kept_views_share_the_fused_buffer()


def test_the_fused_buffer_reading_collects_before_it_clears(loaded_format, monkeypatch):
    """The helper collects garbage before every cache clear of its own, so
    a buffer freed from a reference cycle is dropped, not cached for the
    fusion to reuse (a cycle collected after the last clear failed the
    parent's order at 72,767 > 56,383). The fusion's own clear comes
    after the reading."""
    calls = []
    collect, clear = gc.collect, mx.clear_cache
    fuse = apply_qwen35_moe_gate_up_fusion

    def fusing(model):
        calls.append("fuse")
        return fuse(model)

    monkeypatch.setattr(gc, "collect", lambda: calls.append("collect") or collect())
    monkeypatch.setattr(mx, "clear_cache", lambda: calls.append("clear") or clear())
    monkeypatch.setitem(globals(), "apply_qwen35_moe_gate_up_fusion", fusing)
    _assert_kept_views_share_the_fused_buffer()
    assert calls[:4] == ["collect", "clear", "collect", "clear"]
    assert calls.index("fuse") == 4


def test_a_batched_verify_of_a_fused_format_stack_never_runs_its_views(
    loaded_format, monkeypatch
):
    """mlx-vlm's batched verify (``batch x top_k >= 64``) is not given to
    the original call, which runs the gate and up views (a launch copies a
    view whole) or, at mlx-vlm 0.7.11, each token through the fused stack:
    a fused format stack's goes to the fused verify switch, one launch a
    projection for the block, with the unfused layers' bits."""
    glu = _format_glu(experts=16)
    batch, length, top_k = 8, 2, 8  # 64 routes per position, 8 rows
    mx.random.seed(5)
    x = mx.random.randint(-8, 9, (batch, length, HIDDEN)).astype(mx.float32)
    idx = mx.random.randint(0, 16, shape=(batch, length, top_k)).astype(mx.uint32)
    weights = mx.full((batch, length, top_k), 0.5)
    ref = glu(x, idx, weights)
    mx.eval(ref)
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("experts", glu)]
    assert apply_qwen35_moe_gate_up_fusion(model) == 1
    views, called = (glu.gate_proj, glu.up_proj), []
    call = FormatSwitchLinear.__call__
    monkeypatch.setattr(
        FormatSwitchLinear,
        "__call__",
        lambda self, *a, **k: (called.append(self), call(self, *a, **k))[1],
    )
    out = glu(x, idx, weights)
    mx.eval(out)
    assert mx.array_equal(ref, out).item()
    assert not any(layer is view for layer in called for view in views)
    assert called == [glu.gate_up_proj, glu.down_proj]


@testing.verified("qwen4_exp")
def test_fusing_warm_format_stacks_keeps_the_sorted_tile_route(monkeypatch):
    """On a sorted prefill whose epilogue declines, both fused projections
    run their tile kernel, with the original block's bits."""
    from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

    from omlx.formats import kernels
    from omlx.formats.affine import AffineFormat

    class PackedRows(AffineFormat):
        mode = "test_packed_rows"

        def layout(self, k, n, bits, group_size):
            return {
                "weight": ((n, k * bits // 32), mx.uint32),
                "scales": ((n, k // group_size), mx.bfloat16),
                "biases": ((n, k // group_size), mx.bfloat16),
            }

    fmt = PackedRows()
    formats.load()
    mx.random.seed(615)
    hidden, inter, experts, tokens = 256, 128, 8, 64
    glu = VLMSwitchGLU(hidden, inter, experts)
    for name, k, n in (
        ("gate_proj", hidden, inter),
        ("up_proj", hidden, inter),
        ("down_proj", inter, hidden),
    ):
        weights = (mx.random.normal((experts, n, k)) * 0.125).astype(mx.bfloat16)
        w, scales, biases = mx.quantize(weights, group_size=64, bits=4)
        glu[name] = FormatSwitchLinear(
            fmt, 4, 64, k, {"weight": w, "scales": scales, "biases": biases}
        )
    glu.eval()
    x = mx.random.normal((1, tokens, hidden)).astype(mx.bfloat16)
    idx = mx.random.randint(0, experts, (1, tokens, 2)).astype(mx.uint32)
    weights = mx.full(idx.shape, 0.5, dtype=x.dtype)
    gate, up, down = glu.gate_proj, glu.up_proj, glu.down_proj
    route = (
        "format_switch_tile_nax"
        if format_linear._nax_host()
        else "format_switch_tile_steel"
    )
    called = []
    forward = FormatSwitchLinear.__call__

    def sorted_call(layer, rows, indices, sorted_indices=False):
        assert sorted_indices
        result = forward(layer, rows, indices, sorted_indices=True)
        kept = vars(layer)[format_linear._KEPT]
        tile = kept[3]
        assert tile.proof is kernels.proof(
            route, tile.face, tuple(a.dtype for a in tile.operands)
        )
        signature = (
            rows.shape,
            rows.dtype,
            indices.shape,
            indices.dtype,
            layer.output_dims,
            layer.num_experts,
        )
        assert signature in tile.proof.graphs
        assert not kept[0].proof.ran
        called.append(id(layer))
        return result

    monkeypatch.setattr(FormatSwitchLinear, "__call__", sorted_call)
    monkeypatch.setattr(patch_mod, "fused_gate_up_activation", lambda *a, **kw: None)
    ref = glu(x, idx, weights)
    mx.eval(ref)
    assert sorted(called) == sorted(map(id, (gate, up, down)))
    assert mx.isfinite(ref).all().item() and (ref != 0).any().item()
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("experts", glu)]
    assert apply_qwen35_moe_gate_up_fusion(model) == 1
    fused = glu.gate_up_proj
    assert all(format_linear._KEPT not in vars(layer) for layer in (gate, up, fused))
    assert format_linear._KEPT in vars(down)
    called.clear()
    got = glu(x, idx, weights)
    assert called == [id(fused), id(down)]
    assert mx.array_equal(got.view(mx.uint16), ref.view(mx.uint16)).item()


@testing.verified("qwen4_exp")
def test_gate_and_up_from_two_checkpoint_readers_stay_separate(monkeypatch):
    from copy import copy

    from tests.test_formats_load import _Loadable

    fmt = _Loadable()
    monkeypatch.setitem(formats._FORMATS, "toy", fmt)
    first, second = copy(fmt), copy(fmt)
    glu = SwitchGLU(HIDDEN, INTER, E)
    arrays = {
        "weight": mx.zeros((E, INTER, HIDDEN), mx.uint8),
        "scales": mx.ones((E, INTER), mx.float32),
    }
    glu.gate_proj = FormatSwitchLinear(first, 8, 1, HIDDEN, arrays)
    glu.up_proj = FormatSwitchLinear(second, 8, 1, HIDDEN, arrays)
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("experts", glu)]
    assert apply_qwen35_moe_gate_up_fusion(model) == 0
    assert glu.gate_proj.fmt is first and glu.up_proj.fmt is second


@testing.verified("qwen4_exp")
def test_gate_and_up_require_the_same_input_width():
    from tests.test_formats_load import _Loadable

    class BlockPacked(_Loadable):
        def layout(self, k, n, bits, group_size):
            return {
                "weight": ((n, (k + 31) // 32), mx.uint32),
                "scales": ((n,), mx.float32),
            }

    fmt = BlockPacked()
    glu = SwitchGLU(HIDDEN, INTER, E)
    # Packing pads K to a block. Different input widths have identical storage.
    for name, k in (("gate_proj", HIDDEN), ("up_proj", HIDDEN - 1)):
        tensors = {
            key: mx.zeros((E, *shape), dtype)
            for key, (shape, dtype) in fmt.layout(k, INTER, 4, 32).items()
        }
        glu[name] = FormatSwitchLinear(fmt, 4, 32, k, tensors)
    assert glu.gate_proj.weight.shape == glu.up_proj.weight.shape
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("experts", glu)]
    assert apply_qwen35_moe_gate_up_fusion(model) == 0
    assert glu.gate_proj.input_dims == HIDDEN and glu.up_proj.input_dims == HIDDEN - 1


@testing.verified("qwen4_exp")
def test_gate_and_up_require_a_matching_layout_at_twice_the_output_width():
    from tests.test_formats_load import _Loadable

    class Padded(_Loadable):
        def layout(self, k, n, bits, group_size):
            shape = super().layout(k, n, bits, group_size)
            if n == 2 * INTER:
                shape["weight"] = ((n, k + 32), mx.uint8)
            return shape

    fmt = Padded()
    glu = SwitchGLU(HIDDEN, INTER, E)
    for name in ("gate_proj", "up_proj"):
        tensors = {
            key: mx.zeros((E, *shape), dtype)
            for key, (shape, dtype) in fmt.layout(HIDDEN, INTER, 8, 1).items()
        }
        glu[name] = FormatSwitchLinear(fmt, 8, 1, HIDDEN, tensors)
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("experts", glu)]
    assert apply_qwen35_moe_gate_up_fusion(model) == 0
    assert "gate_up_proj" not in glu


@pytest.mark.parametrize("bad", ["shape", "dtype", "no_layout"])
@testing.verified("qwen4_exp")
def test_equal_gate_and_up_buffers_must_match_the_declared_layout(bad):
    from tests.test_formats_load import _Loadable

    class Reader(_Loadable):
        def layout(self, k, n, bits, group_size):
            if bad == "no_layout":
                return None
            return super().layout(k, n, bits, group_size)

    fmt = Reader()
    glu = SwitchGLU(HIDDEN, INTER, E)
    scales = mx.ones((E, INTER, 1) if bad == "shape" else (E, INTER))
    tensors = {
        "weight": mx.zeros((E, INTER, HIDDEN), mx.uint8),
        "scales": scales.astype(mx.float16 if bad == "dtype" else mx.float32),
    }
    for name in ("gate_proj", "up_proj"):
        glu[name] = FormatSwitchLinear(fmt, 8, 1, HIDDEN, tensors)
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("experts", glu)]
    assert apply_qwen35_moe_gate_up_fusion(model) == 0
    assert "gate_up_proj" not in glu


@pytest.mark.parametrize("lm", [False, True], ids=["vlm", "lm"])
@pytest.mark.parametrize("pending", [None, "gate_proj", "up_proj"])
def test_post_load_fusion_uses_only_materialized_stacks(lm, pending):
    from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

    from tests.test_formats_load import _Loadable

    reader = _Loadable()
    glu = (SwitchGLU if lm else VLMSwitchGLU)(HIDDEN, INTER, E)
    parts = []
    with testing.verified("qwen4_exp"):
        for i, name in enumerate(("gate_proj", "up_proj"), 1):
            arrays = {
                "weight": mx.full((E, INTER, HIDDEN), i, mx.uint8),
                "scales": mx.full((E, INTER), 2 * i, mx.float32),
            }
            bias = mx.full((E, INTER), -i, mx.float32)
            glu[name] = FormatSwitchLinear(
                reader, 8, 1, HIDDEN, arrays, bias, placeholders=name == pending
            )
            parts.append({**arrays, "bias": bias})
    glu.eval()
    gate, up = glu.gate_proj, glu.up_proj
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("experts", glu)]
    # The engine transforms the loaded model after its constructor gate closes.
    with pytest.raises(formats.FormatRefusedError):
        FormatSwitchLinear(reader, 8, 1, HIDDEN, arrays, bias)
    if pending is not None:
        first = type(glu)(HIDDEN, INTER, E)
        model.named_modules = lambda: [("first", first), ("second", glu)]
        with pytest.raises(ValueError, match="placeholder"):
            apply_qwen35_moe_gate_up_fusion(model)
        assert "gate_up_proj" not in first and "gate_up_proj" not in glu
        assert glu.gate_proj is gate and glu.up_proj is up
        with pytest.raises(ValueError, match="placeholder"):
            glu[pending].refuse_placeholders()
        return

    assert apply_qwen35_moe_gate_up_fusion(model) == 1
    fused = glu.gate_up_proj
    assert fused.fmt is reader and not fused.training
    assert (fused.num_experts, fused.input_dims, fused.output_dims) == (
        E,
        HIDDEN,
        2 * INTER,
    )
    fused.refuse_placeholders()
    for name in parts[0]:
        axis = -1 if name == "bias" else 1
        expected = mx.concatenate([p[name] for p in parts], axis)
        assert mx.array_equal(fused[name], expected).item()
    if lm:
        assert "gate_proj" not in glu and "up_proj" not in glu
    else:
        assert glu.gate_proj is gate and glu.up_proj is up
        for layer, part in zip((gate, up), parts):
            assert layer.output_dims == INTER
            for name, expected in part.items():
                assert mx.array_equal(layer[name], expected).item()
