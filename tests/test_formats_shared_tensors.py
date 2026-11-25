# SPDX-License-Identifier: Apache-2.0
"""A format's tensors that one layer keeps once, shared by its whole stack.

``QuantFormat.shared_tensors`` names them: a switch layer holds one sign
vector for all its experts and a gate/up fusion keeps it shared. ``amalgam``
below is every shape rule the framework must honour for them, and
``EncodeContext`` gives a format's writer the layer it is encoding.
"""

from __future__ import annotations

import json

import mlx.core as mx
import pytest

from omlx import formats, oq
from omlx.formats import oqh, testing
from omlx.formats.linear import FormatEmbedding, FormatSwitchLinear
from omlx.patches import qwen35_moe_gate_up
from tests.test_formats_oqh import _group_case


class _Shared(formats.QuantFormat):
    """Codes and a row scale per matrix, and one sign vector per layer."""

    mode = "toy_shared"
    contract = formats.CONTRACT
    codes_name = "codes"
    tensors = ("gains",)
    shared_tensors = ("signs",)

    def layout(self, k, n, bits, group_size):
        if k % 128:
            return None
        return {
            self.codes_name: ((n, k), mx.uint8),
            "gains": ((n,), mx.float32),
            "signs": ((k // 128, 128), mx.bfloat16),
        }

    def decode(self, tensors, bits, group_size):
        signs = tensors["signs"].reshape(-1).astype(mx.float32)
        return (
            (tensors[self.codes_name].astype(mx.float32) - 128)
            * tensors["gains"][..., None]
            * signs
        )


class _NShared(_Shared):
    """A layout whose shared tensor grows with N: not stackable."""

    def layout(self, k, n, bits, group_size):
        one = super().layout(k, n, bits, group_size)
        one["signs"] = ((n,), mx.bfloat16)
        return one


def _tensors(fmt, experts, n, k, seed=0):
    mx.random.seed(seed)
    return {
        fmt.codes_name: mx.random.randint(0, 256, (experts, n, k)).astype(mx.uint8),
        "gains": mx.ones((experts, n), mx.float32),
        "signs": mx.ones((k // 128, 128), mx.bfloat16),
    }


@pytest.fixture
def registered(monkeypatch):
    formats.load()
    monkeypatch.setattr(formats, "_FORMATS", dict(formats._FORMATS))
    fmt = _Shared()
    formats.register(fmt)
    return fmt


def test_a_shared_tensor_is_one_layer_s_however_many_matrices_it_serves(registered):
    fmt = registered
    assert formats.all_tensor_names(fmt) == ("codes", "gains", "signs")
    with testing.verified("example"):
        layer = FormatSwitchLinear(fmt, 8, 1, 256, _tensors(fmt, 4, 8, 256))
    assert layer["signs"].shape == (2, 128)
    assert layer._tensors == ("codes", "gains")
    assert layer._shared_tensors == ("signs",)
    # The rows rule holds: two matrices stack along N, the sign vector does not.
    assert formats.stackable(fmt, 256, 8, 8, 1)
    with testing.verified("example"):
        other = FormatSwitchLinear(fmt, 8, 1, 256, _tensors(fmt, 4, 8, 256, seed=2))
        stacked = layer._stacked_outputs(other)
    assert stacked["codes"].shape == (4, 16, 256)
    assert stacked["gains"].shape == (4, 16)
    assert stacked["signs"] is layer["signs"]
    assert formats.tensor_names(stacked) == ("codes", "gains")


def test_a_layout_whose_shared_tensor_grows_with_n_is_not_stackable():
    fmt = _NShared()
    assert not formats.stackable(fmt, 256, 8, 8, 1)
    # Its N-tensor still lays a matrix out; only the rows rule fails.
    assert formats.check_layout(fmt, 8, 256, fmt.layout(256, 8, 8, 1)) is None


def test_register_refuses_malformed_shared_tensors(monkeypatch):
    formats.load()
    monkeypatch.setattr(formats, "_FORMATS", dict(formats._FORMATS))

    class Overlap(_Shared):
        tensors = ("gains", "signs")

    for shared, match in [
        (("signs", "signs"), "each named once"),
        (("codes",), "codes"),
        (("weight",), "weight"),
        (("gains",), "codes|tensors"),
    ]:
        fmt = Overlap()
        fmt.shared_tensors = shared
        with pytest.raises(ValueError, match=match):
            formats.register(fmt)
        assert formats.get(fmt.mode) is None


def test_a_shared_tensor_is_held_whole_by_a_stack_and_gate_up_fusion_keeps_it(
    registered,
):
    from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

    fmt = registered
    glu = VLMSwitchGLU(256, 256, 4)
    with testing.verified("example"):
        glu.gate_proj = FormatSwitchLinear(fmt, 8, 1, 256, _tensors(fmt, 4, 256, 256))
        glu.up_proj = FormatSwitchLinear(
            fmt, 8, 1, 256, _tensors(fmt, 4, 256, 256, seed=2)
        )
        glu.up_proj["signs"] = glu.gate_proj["signs"]
        glu.down_proj = FormatSwitchLinear(
            fmt, 8, 1, 256, _tensors(fmt, 4, 256, 256, seed=3)
        )
    assert qwen35_moe_gate_up._can_fuse(glu)
    qwen35_moe_gate_up._fuse_one(glu)
    fused = glu.gate_up_proj
    # A VLM switch keeps its gate and up views, and they read the one vector.
    assert fused["signs"] is glu.gate_proj["signs"] is glu.up_proj["signs"]
    assert fused["signs"].shape == (2, 128)
    assert fused["codes"].shape == (4, 512, 256)
    assert glu.gate_proj["codes"].shape == glu.up_proj["codes"].shape == (4, 256, 256)


def test_a_shared_tensor_that_differs_refuses_the_fusion(registered):
    fmt = registered
    with testing.verified("example"):
        gate = FormatSwitchLinear(fmt, 8, 1, 256, _tensors(fmt, 4, 64, 256))
        up = FormatSwitchLinear(fmt, 8, 1, 256, _tensors(fmt, 4, 64, 256, seed=2))
        down = FormatSwitchLinear(fmt, 8, 1, 256, _tensors(fmt, 4, 64, 256))
    up["signs"] = mx.random.normal(up["signs"].shape).astype(mx.bfloat16)
    mx.eval(gate.parameters(), up.parameters())
    glu = type(
        "Glu",
        (),
        dict(gate_proj=gate, up_proj=up, down_proj=down),
    )()
    assert not qwen35_moe_gate_up._can_fuse(glu)


def test_encode_layer_is_the_only_writer_hook(registered):
    """Rule 26: there is no ``encode`` to fall back to. A format that lays a
    matrix out but implements no ``encode_layer`` cannot write it, and the
    framework raises rather than silently keeping it affine."""
    assert not hasattr(formats.QuantFormat, "encode")

    class NoWriter(_Shared):
        mode = "toy_shared_no_writer"

    fmt = NoWriter()
    formats.register(fmt)
    w = mx.zeros((8, 256), mx.float32)
    with pytest.raises(NotImplementedError, match="encode_layer|does not encode"):
        oq._encode_in_format(fmt, "model.layers.0.mlp.gate_proj.weight", w, None, 8, 1)


def test_encode_context_names_the_layer_a_format_writes(registered, monkeypatch):
    fmt = registered
    seen = []

    class Contextual(_Shared):
        def encode_layer(self, w, importance, ctx):
            seen.append(ctx)
            n, k = w.shape[-2:]
            return {
                self.codes_name: mx.full((*w.shape[:-2], n, k), 128, mx.uint8),
                "gains": mx.ones((*w.shape[:-2], n), mx.float32),
                "signs": mx.ones((k // 128, 128), mx.bfloat16),
            }

    fmt = Contextual()
    fmt.mode = "toy_shared_context"
    formats.register(fmt)
    config = {"model_type": "qwen3_5_moe"}
    w = mx.zeros((4, 8, 256), mx.float32)
    importance = mx.zeros((4, 256), mx.float32)
    encoded = oq._encode_in_format(
        fmt,
        "model.layers.0.mlp.switch_mlp.gate_proj.weight",
        w,
        importance,
        8,
        1,
        config=config,
    )
    assert set(encoded) == {"codes", "gains", "signs"}
    assert encoded["codes"].shape == (4, 8, 256)
    assert encoded["signs"].shape == (2, 128)
    (ctx,) = seen
    assert ctx.path == "model.layers.0.mlp.switch_mlp.gate_proj.weight"
    assert ctx.is_switch and ctx.bits == 8 and ctx.group_size == 1
    assert ctx.config == config
    assert json.dumps(ctx.config) == json.dumps(config)


def test_format_embedding_as_linear_keeps_shared_tensors(registered):
    fmt = registered
    tensors = {
        fmt.codes_name: mx.random.randint(0, 256, (8, 256)).astype(mx.uint8),
        "gains": mx.ones((8,), mx.float32),
        "signs": mx.ones((2, 128), mx.bfloat16),
    }
    with testing.verified("example"):
        emb = FormatEmbedding(fmt, 8, 1, 256, tensors)
    x = mx.ones((1, 256), mx.float32)
    got = emb.as_linear(x)
    want = x @ emb._decoded().T
    assert mx.allclose(got, want, atol=1e-4, rtol=1e-4).item()


def test_accept_cannot_change_shared_tensors(registered):
    class Mutating(_Shared):
        mode = "toy_shared_mutating"

        def accept(self, record):
            self.shared_tensors = ("weight",)
            return self

    fmt = Mutating()
    formats.register(fmt)
    with pytest.raises(ValueError):
        oq._format_reader(fmt, fmt.record())


def test_stackable_accepts_scalar_and_list_shared_shapes():
    class ScalarShared(_Shared):
        mode = "toy_scalar_shared"
        shared_tensors = ("seed",)

        def layout(self, k, n, bits, group_size):
            if k % 128:
                return None
            return {
                self.codes_name: ((n, k), mx.uint8),
                "gains": ((n,), mx.float32),
                "seed": ((), mx.uint32),
            }

    class ListShared(_Shared):
        mode = "toy_list_shared"
        shared_tensors = ("signs",)

        def layout(self, k, n, bits, group_size):
            if k % 128:
                return None
            return {
                self.codes_name: ((n, k), mx.uint8),
                "gains": ((n,), mx.float32),
                "signs": ([k // 128, 128], mx.bfloat16),
            }

    assert formats.stackable(ScalarShared(), 256, 8, 8, 1)
    assert formats.stackable(ListShared(), 256, 8, 8, 1)


def test_rows_stack_refuses_differing_shared_tensors():
    class IgnoredSigns(_Shared):
        mode = "toy_ignored_signs"

        def decode(self, tensors, bits, group_size):
            return (tensors[self.codes_name].astype(mx.float32) - 128) * tensors[
                "gains"
            ][..., None]

    fmt = IgnoredSigns()
    first = {
        fmt.codes_name: mx.full((8, 256), 129, mx.uint8),
        "gains": mx.ones((8,), mx.float32),
        "signs": mx.ones((2, 128), mx.bfloat16),
    }
    second = {
        fmt.codes_name: mx.full((8, 256), 129, mx.uint8),
        "gains": mx.ones((8,), mx.float32),
        "signs": -mx.ones((2, 128), mx.bfloat16),
    }
    with pytest.raises(AssertionError, match="shared tensor"):
        testing.rows_stack(fmt, first, second, 256, 8, 1)


def test_encode_in_format_shared_tensor_buffer_budget_not_multiplied_by_experts(
    registered, monkeypatch
):
    class BigShared(_Shared):
        mode = "toy_big_shared"

        def layout(self, k, n, bits, group_size):
            if k % 128:
                return None
            return {
                self.codes_name: ((n, k), mx.uint8),
                "gains": ((n,), mx.float32),
                "signs": ((k // 128, 128), mx.float32),
            }

        def encode_layer(self, w, importance, ctx):
            n, k = w.shape[-2:]
            return {
                self.codes_name: mx.full((*w.shape[:-2], n, k), 128, mx.uint8),
                "gains": mx.ones((*w.shape[:-2], n), mx.float32),
                "signs": mx.ones((k // 128, 128), mx.float32),
            }

    fmt = BigShared()
    formats.register(fmt)
    # 4 experts, n=1, k=256: codes=1024 B stacked, signs=1024 B unstacked
    # (4 * 1024 = 4096 if wrongly multiplied by experts).
    monkeypatch.setattr(oq, "_METAL_MAX_BUFFER", 1500)
    w = mx.zeros((4, 1, 256), mx.bfloat16)
    encoded = oq._encode_in_format(fmt, "e.weight", w, mx.ones((256,)), 8, 1)
    assert encoded is not None
    assert encoded["signs"].shape == (2, 128)


def test_qwen4_ple_fuse_format_shards_keeps_shared_tensors():
    from omlx.formats.linear import FormatEmbedding
    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import ShardedEmbedding

    def shard(seed):
        tensors, _ = _group_case(128, 8, 4, 128, seed=seed)
        with testing.verified("example"):
            return FormatEmbedding(oqh.OQH_AFFINE, 4, 128, 128, tensors)

    holder = ShardedEmbedding(16, 128, 2)
    holder.shards = [shard(0), shard(0)]
    expected_signs = holder.shards[0]["oqh_signs"]
    assert holder.fuse_quantized_shards() is True
    assert holder.fused["oqh_signs"].shape == (1, 128)
    assert mx.array_equal(holder.fused["oqh_signs"], expected_signs).item()

    bad = ShardedEmbedding(16, 128, 2)
    a, b = shard(0), shard(0)
    b["oqh_signs"] = mx.random.normal(b["oqh_signs"].shape).astype(mx.bfloat16)
    bad.shards = [a, b]
    assert bad.fuse_quantized_shards() is False


def test_layer_by_layer_releases_shared_tensors():
    from omlx import oq_verify
    from omlx.formats.linear import FormatLinear

    tensors, _ = _group_case(128, 8, 4, 128)
    with testing.verified("example"):
        layer = FormatLinear(oqh.OQH_AFFINE, 4, 128, 128, tensors)
    x = mx.random.normal((1, 128)).astype(mx.bfloat16)
    with oq_verify.layer_by_layer([layer]):
        mx.eval(layer(x))
    assert layer["oqh_signs"].size == 0


def test_list_shared_shape_encodes_and_fuses(registered):
    class ListShared(_Shared):
        mode = "toy_shared_list"

        def layout(self, k, n, bits, group_size):
            one = super().layout(k, n, bits, group_size)
            one["signs"] = ([k // 128, 128], mx.bfloat16)
            return one

    fmt = ListShared()
    formats.register(fmt)
    w = mx.zeros((4, 8, 256), mx.float32)
    seen = []

    class Writer(ListShared):
        def encode_layer(self, w, importance, ctx):
            seen.append(ctx)
            return {
                "codes": mx.zeros((*w.shape[:-2], w.shape[-2], w.shape[-1]), mx.uint8),
                "gains": mx.ones((*w.shape[:-2], w.shape[-2]), mx.float32),
                "signs": mx.ones((2, 128), mx.bfloat16),
            }

    writer = Writer()
    writer.mode = "toy_shared_list_writer"
    formats.register(writer)
    out = oq._encode_in_format(
        writer,
        "model.layers.0.mlp.gate_proj.weight",
        w,
        mx.ones((4, 256), mx.float32),
        8,
        1,
    )
    assert out is not None and out["signs"].shape == (2, 128)

    with testing.verified("example"):
        gate = FormatSwitchLinear(fmt, 8, 1, 256, _tensors(fmt, 4, 256, 256))
        up = FormatSwitchLinear(fmt, 8, 1, 256, _tensors(fmt, 4, 256, 256))
        down = FormatSwitchLinear(fmt, 8, 1, 256, _tensors(fmt, 4, 256, 256))
    up["signs"] = gate["signs"]
    glu = type("Glu", (), dict(gate_proj=gate, up_proj=up, down_proj=down))()
    assert qwen35_moe_gate_up._can_fuse(glu)


def test_encode_context_config_is_isolated(registered):
    class MutatingWriter(_Shared):
        mode = "toy_shared_mutating_cfg"

        def encode_layer(self, w, importance, ctx):
            ctx.config["model_type"] = "bad"
            ctx.config["text_config"]["k"] = 9
            return {
                "codes": mx.zeros((*w.shape[:-2], w.shape[-2], w.shape[-1]), mx.uint8),
                "gains": mx.ones((*w.shape[:-2], w.shape[-2]), mx.float32),
                "signs": mx.ones((w.shape[-1] // 128, 128), mx.bfloat16),
            }

    writer = MutatingWriter()
    formats.register(writer)
    cfg = {"model_type": "qwen4_exp", "text_config": {"k": 1}}
    out = oq._encode_in_format(
        writer,
        "model.layers.0.mlp.gate_proj.weight",
        mx.zeros((8, 256), mx.float32),
        mx.ones((256,), mx.float32),
        8,
        1,
        config=cfg,
    )
    assert out is not None
    assert cfg == {"model_type": "qwen4_exp", "text_config": {"k": 1}}


def test_encode_layer_gets_fresh_config_deepcopy_per_chunk_run(registered, monkeypatch):
    calls = 0

    class ChunkMutatingWriter(_Shared):
        mode = "toy_shared_chunk_cfg"

        def layout(self, k, n, bits, group_size):
            return {
                self.codes_name: ((n, k), mx.uint8),
                "gains": ((n,), mx.float32),
                "signs": ((1, k), mx.bfloat16),
            }

        def encode_layer(self, w, importance, context):
            nonlocal calls
            calls += 1
            assert context.config["k"] == []
            context.config["k"].append(1)
            n, k = w.shape[-2:]
            return {
                self.codes_name: mx.zeros((*w.shape[:-2], n, k), mx.uint8),
                "gains": mx.ones((*w.shape[:-2], n), mx.float32),
                "signs": mx.ones((1, k), mx.bfloat16),
            }

    writer = ChunkMutatingWriter()
    formats.register(writer)
    monkeypatch.setattr(oq, "_LAUNCH_WEIGHTS", 8 * 32)
    cfg = {"k": []}
    out = oq._encode_in_format(
        writer,
        "model.layers.0.mlp.switch_mlp.gate_proj.weight",
        mx.zeros((2, 8, 32), mx.float32),
        mx.ones((2, 32), mx.float32),
        8,
        1,
        config=cfg,
    )
    assert out is not None
    assert out["codes"].shape == (2, 8, 32)
    assert calls == 2
    assert cfg == {"k": []}


def test_refine_preserves_shared_tensors_on_switch(registered, tmp_path):
    class RefiningShared(_Shared):
        mode = "toy_shared_refine"

        def __init__(self, mismatch=False):
            super().__init__()
            self.mismatch = mismatch
            self.calls = 0

        def refine(self, w, tensors, hessian, bits, group_size):
            self.calls += 1
            signs = tensors["signs"]
            if self.mismatch and self.calls == 2:
                signs = -signs
            return {
                "codes": mx.full(tensors["codes"].shape, 128, dtype=mx.uint8),
                "gains": mx.ones(tensors["gains"].shape, dtype=mx.float32),
                "signs": signs,
            }

    fmt = RefiningShared()
    quant = {"layers.0.mlp.gate_proj": {"mode": fmt.mode, "bits": 8, "group_size": 1}}
    wmap_no_shared = {
        "layers.0.mlp.gate_proj.codes": "s.safetensors",
        "layers.0.mlp.gate_proj.gains": "s.safetensors",
    }
    writer_missing = oq._OQRefineWriter(tmp_path, wmap_no_shared, quant, quant_fmt=fmt)
    assert not writer_missing.is_format("layers.0.mlp.gate_proj")

    wmap = {**wmap_no_shared, "layers.0.mlp.gate_proj.signs": "s.safetensors"}
    writer = oq._OQRefineWriter(tmp_path, wmap, quant, quant_fmt=fmt)
    assert writer._parts() == ("codes", "gains", "signs")
    assert writer.is_format("layers.0.mlp.gate_proj")

    w = mx.zeros((4, 8, 256), mx.float32)
    payload = _tensors(fmt, 4, 8, 256, seed=1)
    gram = mx.eye(256, dtype=mx.float32)
    err = oq._format_hessian_error(fmt, w, payload, gram, 8, 1)
    assert err > 0.0

    class _Collector:
        def expert_ranges(self, owner, n_exp, in_dim, out_dim):
            return [(0, 2), (2, 4)]

        def switch_gram(self, owner, first, stop):
            return mx.broadcast_to(gram, (stop - first, 256, 256))

    targets = [("layers.0.mlp.gate_proj", w, "format", payload, 8, 1)]
    res = oq._refine_switch_codes(
        _Collector(), "layers.0.mlp.gate_proj", targets, writer
    )
    assert res[0] is not None
    refined, before, after = res[0]
    assert after <= before
    assert refined["codes"].shape == (4, 8, 256)
    assert refined["signs"].shape == (2, 128)
    assert mx.array_equal(refined["signs"], payload["signs"]).item()

    bad_fmt = RefiningShared(mismatch=True)
    bad_writer = oq._OQRefineWriter(tmp_path, wmap, quant, quant_fmt=bad_fmt)
    with pytest.raises(ValueError, match="shared tensor"):
        oq._refine_switch_codes(
            _Collector(), "layers.0.mlp.gate_proj", targets, bad_writer
        )


def test_stacked_outputs_refuses_shared_tensor_dtype_mismatch(registered):
    class _AnyFloatShared(_Shared):
        mode = "toy_shared_any_float"

        def layout(self, k, n, bits, group_size):
            one = super().layout(k, n, bits, group_size)
            if one is not None:
                one["signs"] = ((k // 128, 128), None)
            return one

    fmt = _AnyFloatShared()
    formats.register(fmt)
    ta = _tensors(fmt, 4, 8, 256, seed=1)
    tb = _tensors(fmt, 4, 8, 256, seed=2)
    ta["signs"] = mx.ones((2, 128), mx.bfloat16)
    tb["signs"] = mx.ones((2, 128), mx.float16)
    with testing.verified("example"):
        la = FormatSwitchLinear(fmt, 8, 1, 256, ta)
        lb = FormatSwitchLinear(fmt, 8, 1, 256, tb)
    with pytest.raises(ValueError, match="shared tensor"):
        la._stacked_outputs(lb)


def test_format_hessian_error_inverts_the_input_transform():
    from omlx.formats import transforms
    from tests.test_formats_transforms import _Rotated

    class _RefiningRotated(_Rotated):
        def __init__(self, target):
            super().__init__()
            self.target = target

        def refine(self, w, tensors, hessian, bits, group_size):
            return {"codes": self.target, "signs": tensors["signs"]}

    fmt = _Rotated()
    k, n = 256, 8
    mx.random.seed(7)
    plain = mx.random.normal((n, k)).astype(mx.float32) * 0.1
    signs = mx.ones((k // 128, 128), mx.float32)
    tx = transforms.SignedWhtTransform(block=128, tensors=("signs",))
    exact = tx.apply(plain, signs)
    noisy = exact + 0.1 * (plain - exact)
    tensors = {"codes": exact, "signs": signs}
    hessian = mx.eye(k)
    before = oq._format_hessian_error(fmt, plain, tensors, hessian, 8, 1)
    noisy_tensors = {"codes": noisy, "signs": signs}
    after = oq._format_hessian_error(fmt, plain, noisy_tensors, hessian, 8, 1)
    assert before < 1e-6
    assert after > before
    ref2 = oq._refine_format_tensors(
        _RefiningRotated(exact), fmt, "w", plain, noisy_tensors, hessian, 8, 1
    )
    assert ref2 is not None and ref2[2] < 1e-6 < ref2[1]

    plain3 = mx.random.normal((2, n, k)).astype(mx.float32) * 0.1
    exact3 = tx.apply(plain3, signs)
    noisy3 = exact3 + 0.1 * (plain3 - exact3)
    hessian3 = mx.broadcast_to(hessian, (2, k, k))
    before3 = oq._format_hessian_error(
        fmt, plain3, {"codes": exact3, "signs": signs}, hessian3, 8, 1
    )
    noisy3_tensors = {"codes": noisy3, "signs": signs}
    after3 = oq._format_hessian_error(fmt, plain3, noisy3_tensors, hessian3, 8, 1)
    assert before3 < 1e-6
    assert after3 > before3
    ref3 = oq._refine_format_tensors(
        _RefiningRotated(exact3), fmt, "w3", plain3, noisy3_tensors, hessian3, 8, 1
    )
    assert ref3 is not None and ref3[2] < 1e-6 < ref3[1]


def test_configured_format_shared_tensors_mismatch_is_named(registered, tmp_path):
    other = _Shared()
    other.shared_tensors = ("signs", "extra")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))
    with pytest.raises(ValueError, match=r"shared \('signs', 'extra'\)"):
        oq._open_oq_run(
            str(tmp_path),
            str(tmp_path / "out"),
            4,
            enhanced=True,
            quant_format=other,
        )
