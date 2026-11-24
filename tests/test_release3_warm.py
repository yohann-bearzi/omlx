# SPDX-License-Identifier: Apache-2.0
"""Release-3 load-time warming and zero-new-proofs serving tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.formats import affine, kernels, linear, testing
from omlx.formats.linear import FormatLinear, FormatSwitchLinear
from omlx.patches import (
    format_load,
    moe_verify_gather,
    qwen35_moe_gate_up,
    qwen35_moe_routed_decode,
    qwen35_verify_qmm,
)


class _Future:
    def __init__(self, error=None):
        self.error = error
        self.awaited = 0

    def result(self, timeout=None):
        self.awaited += 1
        if self.error is not None:
            raise self.error


def _write_qwen_config(path, dtype="bfloat16"):
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {
                    "model_type": "qwen4_exp_text",
                    "dtype": dtype,
                },
            }
        )
    )


def _quantized_format_linear(fmt, n, k, bits, group_size, dtype, *, seed=0):
    mx.random.seed(seed)
    w_fp = mx.random.normal((n, k), dtype=dtype) * mx.array(0.02, dtype=dtype)
    w, scales, biases = mx.quantize(w_fp, group_size=group_size, bits=bits)
    mx.eval(w, scales, biases)
    layout = fmt.layout(k, n, bits, group_size)
    return (
        FormatLinear(
            fmt,
            bits,
            group_size,
            k,
            {"weight": w, "scales": scales, "biases": biases},
            None,
            placeholders=False,
            _load_layout=layout,
        ),
        layout,
    )


def _quantized_format_switch(fmt, experts, n, k, bits, group_size, dtype, *, seed=0):
    mx.random.seed(seed)
    w_fp = mx.random.normal((experts, n, k), dtype=dtype) * mx.array(0.02, dtype=dtype)
    w, scales, biases = mx.quantize(w_fp, group_size=group_size, bits=bits)
    mx.eval(w, scales, biases)
    layout = fmt.layout(k, n, bits, group_size)
    return (
        FormatSwitchLinear(
            fmt,
            bits,
            group_size,
            k,
            {"weight": w, "scales": scales, "biases": biases},
            None,
            placeholders=False,
            _load_layout=layout,
        ),
        layout,
    )


def test_proof_count_tracks_cold_prove_only(monkeypatch):
    """kernels.proof_count() advances on cold Proof.prove() and not on warm hits."""
    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    monkeypatch.setattr(kernels, "_PROOFS", {})
    before = kernels.proof_count()
    proof = kernels.proof("count_test", "k")

    def build():
        x = mx.zeros((8,), dtype=mx.float32, stream=mx.gpu)
        return (x,), lambda a: a + 1

    import logging

    log = logging.getLogger("test_proof_count")
    assert proof.prove("m1", build, log, "engaged", "failed") is True
    assert kernels.proof_count() == before + 1
    assert proof.prove("m1", build, log, "engaged", "failed") is True
    assert kernels.proof_count() == before + 1
    assert proof.prove("m2", build, log, "engaged", "failed") is True
    assert kernels.proof_count() == before + 2


def test_warm_layer_checkpoint_dtype_only(monkeypatch, tmp_path):
    """A bfloat16 checkpoint warms only bfloat16 Metals and never float16."""
    _write_qwen_config(tmp_path, dtype="bfloat16")
    fmt = affine.AffineFormat()
    layout_dense = fmt.layout(256, 32, 8, 64)
    tensors_dense = {
        name: mx.zeros(shape, mx.float32 if dtype is None else dtype)
        for name, (shape, dtype) in layout_dense.items()
    }
    layout_switch = fmt.layout(256, 64, 4, 64)
    tensors_switch = {
        name: mx.zeros((8, *shape), mx.float32 if dtype is None else dtype)
        for name, (shape, dtype) in layout_switch.items()
    }

    warm_instances = []
    warm_proof_metals = []

    def spy_warm(instances):
        warm_instances.extend(instances.keys())
        return (_Future(),)

    def spy_warm_proofs(tasks):
        for _, metal, _ in tasks:
            warm_proof_metals.append(metal)
        return (_Future(),)

    monkeypatch.setattr(kernels, "warm", spy_warm)
    monkeypatch.setattr(kernels, "warm_proofs", spy_warm_proofs)

    with format_load.loading(tmp_path) as checkpoint:
        checkpoint.verified = True
        checkpoint.verified_modes = frozenset({"affine"})
        dense = FormatLinear(
            fmt, 8, 64, 256, tensors_dense, None, _load_layout=layout_dense
        )
        switch = FormatSwitchLinear(
            fmt, 4, 64, 256, tensors_switch, None, _load_layout=layout_switch
        )
        format_load._warm_layer(
            dense, fmt, layout_dense, 256, tensors_dense, checkpoint
        )
        format_load._warm_layer(
            switch, fmt, layout_switch, 256, tensors_switch, checkpoint
        )

    assert warm_instances, "expected FormatLinear/FormatSwitchLinear Metal instances"
    for inst in warm_instances:
        for dtype, _ in inst.inputs:
            assert dtype != mx.float16, f"unexpected float16 input in {inst}"
        for dtype in inst.outputs:
            assert dtype != mx.float16, f"unexpected float16 output in {inst}"
        for _, _, val in inst.template:
            assert val != mx.float16, f"unexpected float16 template in {inst}"
    for metal in warm_proof_metals:
        assert mx.float16 not in (metal if isinstance(metal, tuple) else (metal,))


def test_warm_fails_closed_on_build_and_plan_errors(monkeypatch, tmp_path):
    """Load warming fails closed: build errors and plan errors fail the load."""
    checkpoint = format_load._Checkpoint(tmp_path)
    trailing = _Future()
    checkpoint.warmers.extend((_Future(RuntimeError("metal compile error")), trailing))
    with pytest.raises(RuntimeError, match="metal compile error"):
        checkpoint.join_warmers()
    assert trailing.awaited == 1
    assert checkpoint.warmers == []

    # A warmer failure when the loading() body succeeds fails the load.
    with (
        pytest.raises(RuntimeError, match="bg build boom"),
        format_load.loading(tmp_path) as cp,
    ):
        cp.warmers.append(_Future(RuntimeError("bg build boom")))

    # A failing loading() body still awaits warmers without masking its own error.
    trailing2 = _Future(RuntimeError("secondary warm error"))
    with (
        pytest.raises(ValueError, match="primary load error"),
        format_load.loading(tmp_path) as cp,
    ):
        cp.warmers.append(trailing2)
        raise ValueError("primary load error")
    assert trailing2.awaited == 1

    # A planning error in _warm_layer propagates instead of being swallowed.
    monkeypatch.setattr(
        format_load,
        "warm_layer",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("plan boom")),
    )
    cp2 = format_load._Checkpoint(tmp_path)
    built = SimpleNamespace(output_dims=32, bits=4, group_size=64)
    with pytest.raises(RuntimeError, match="plan boom"):
        format_load._warm_layer(built, SimpleNamespace(mode="toy"), {}, 256, {}, cp2)


def test_warm_moe_model_fails_closed(monkeypatch):
    """Post-load MoE warm fails closed when a background warm future raises."""
    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP
    from mlx_vlm.models.switch_layers import SwiGLU, SwitchGLU

    if not mx.metal.is_available():
        pytest.skip("requires Metal")

    fmt = affine.AffineFormat()
    dtype = mx.bfloat16
    hidden, inter, experts, top_k = 512, 256, 128, 10

    class _MiniBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = experts
            self.top_k = top_k
            self.gate = nn.Linear(hidden, experts, bias=False)
            self.gate.weight = mx.zeros((experts, hidden), dtype=dtype)
            self.switch_mlp = SwitchGLU.__new__(SwitchGLU)
            nn.Module.__init__(self.switch_mlp)
            self.switch_mlp.activation = SwiGLU()
            self.switch_mlp.gate_up_proj, _ = _quantized_format_switch(
                fmt, experts, 2 * inter, hidden, 4, 64, dtype, seed=1
            )
            self.switch_mlp.down_proj, _ = _quantized_format_switch(
                fmt, experts, hidden, inter, 4, 64, dtype, seed=2
            )
            self.shared_expert = Qwen3_5MLP.__new__(Qwen3_5MLP)
            nn.Module.__init__(self.shared_expert)
            self.shared_expert.gate_proj, _ = _quantized_format_linear(
                fmt, inter, hidden, 8, 64, dtype, seed=3
            )
            self.shared_expert.up_proj, _ = _quantized_format_linear(
                fmt, inter, hidden, 8, 64, dtype, seed=4
            )
            self.shared_expert.down_proj, _ = _quantized_format_linear(
                fmt, hidden, inter, 8, 64, dtype, seed=5
            )
            self.shared_expert_gate, _ = _quantized_format_linear(
                fmt, 1, hidden, 8, 64, dtype, seed=6
            )

    with testing.verified("qwen4_exp"):
        block = _MiniBlock()
        block.eval()

    monkeypatch.setattr(
        kernels,
        "warm_proofs",
        lambda tasks: (_Future(RuntimeError("moe warm boom")),),
    )
    with pytest.raises(RuntimeError, match="moe warm boom"):
        qwen35_moe_routed_decode.warm_moe_model(block, mtp_enabled=False)


def test_zero_new_proofs_on_first_served_request_after_load(monkeypatch, tmp_path):
    """After load-time warm + post-load MoE warm, first served calls compile 0 proofs."""
    if not mx.metal.is_available():
        pytest.skip("requires Metal")

    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP
    from mlx_vlm.models.switch_layers import SwiGLU
    from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

    _write_qwen_config(tmp_path, dtype="bfloat16")
    monkeypatch.setattr(kernels, "_PROOFS", {})
    monkeypatch.setattr(kernels, "_RAW_KERNELS", {})

    fmt = affine.AffineFormat()
    dtype = mx.bfloat16
    hidden, inter, experts, top_k = 512, 256, 128, 10

    class _MiniModel(nn.Module):
        def __init__(self, checkpoint):
            super().__init__()
            # Dense FormatLinear layers: aligned large, aligned small, unaligned N=1 & N=4
            self.proj_aligned, lay_a = _quantized_format_linear(
                fmt, 256, hidden, 8, 64, dtype, seed=10
            )
            self.proj_wide_n, lay_w = _quantized_format_linear(
                fmt, 512, hidden, 8, 64, dtype, seed=11
            )
            self.proj_n1, lay_n1 = _quantized_format_linear(
                fmt, 1, hidden, 8, 64, dtype, seed=12
            )
            self.proj_n4, lay_n4 = _quantized_format_linear(
                fmt, 4, hidden, 8, 64, dtype, seed=13
            )

            # MoE block matching Qwen3_5MoeSparseMoeBlock layout
            self.moe = nn.Module()
            self.moe.num_experts = experts
            self.moe.top_k = top_k
            self.moe.gate = nn.Linear(hidden, experts, bias=False)
            self.moe.gate.weight = mx.random.normal(
                (experts, hidden), dtype=dtype
            ) * mx.array(0.05, dtype=dtype)
            mx.eval(self.moe.gate.weight)

            sw = VLMSwitchGLU.__new__(VLMSwitchGLU)
            nn.Module.__init__(sw)
            sw.activation = SwiGLU()
            sw.gate_proj, lay_g = _quantized_format_switch(
                fmt, experts, inter, hidden, 4, 64, dtype, seed=20
            )
            sw.up_proj, lay_u = _quantized_format_switch(
                fmt, experts, inter, hidden, 4, 64, dtype, seed=21
            )
            sw.down_proj, lay_d = _quantized_format_switch(
                fmt, experts, hidden, inter, 4, 64, dtype, seed=22
            )
            self.moe.switch_mlp = sw

            sh = Qwen3_5MLP.__new__(Qwen3_5MLP)
            nn.Module.__init__(sh)
            sh.gate_proj, lay_sg = _quantized_format_linear(
                fmt, inter, hidden, 8, 64, dtype, seed=30
            )
            sh.up_proj, lay_su = _quantized_format_linear(
                fmt, inter, hidden, 8, 64, dtype, seed=31
            )
            sh.down_proj, lay_sd = _quantized_format_linear(
                fmt, hidden, inter, 8, 64, dtype, seed=32
            )
            self.moe.shared_expert = sh
            self.moe.shared_expert_gate = self.proj_n1

            # Trigger load-time warming just as format_load._format_layer does.
            for mod, lay in (
                (self.proj_aligned, lay_a),
                (self.proj_wide_n, lay_w),
                (self.proj_n1, lay_n1),
                (self.proj_n4, lay_n4),
                (sw.gate_proj, lay_g),
                (sw.up_proj, lay_u),
                (sw.down_proj, lay_d),
                (sh.gate_proj, lay_sg),
                (sh.up_proj, lay_su),
                (sh.down_proj, lay_sd),
            ):
                tensors = {n: mod[n] for n in mod._tensors}
                format_load._warm_layer(
                    mod, fmt, lay, mod.input_dims, tensors, checkpoint
                )

    with format_load.loading(tmp_path) as checkpoint:
        checkpoint.verified = True
        checkpoint.verified_modes = frozenset({"affine"})
        model = _MiniModel(checkpoint)
        model.eval()
    # loading() joined all background warmers here.

    # Post-load gate+up fusion and MoE warm (as engine load does).
    qwen35_moe_gate_up._fuse_one(model.moe.switch_mlp)
    qwen35_moe_routed_decode.warm_moe_model(model, mtp_enabled=True)

    # Snapshot proof_count BEFORE any served request runs.
    before = kernels.proof_count()

    # 1. FormatLinear: 1-row, unarmed tile_wide (V=2,3,4,5 & PARTIAL_M False/True),
    #    armed verify (_shared_rows), prefill (tile_dense), and unaligned N=1/N=4.
    for mod in (model.proj_aligned, model.proj_wide_n, model.proj_n1, model.proj_n4):
        for rows in (1, 2, 3, 4, 5, 7, 8, 9, 12, 16, 32, 64):
            x = mx.zeros((1, rows, hidden), dtype=dtype)
            y = mod(x)
            mx.eval(y)
        qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=True)
        try:
            for rows in (2, 3, 4):
                x = mx.zeros((1, rows, hidden), dtype=dtype)
                y = mod(x)
                mx.eval(y)
        finally:
            qwen35_verify_qmm.set_verify_qmm_armed(False, row_exact=False)

    # 2. FormatSwitchLinear: rows (per_row=1 & per_row=top_k) and sorted tile_gather
    #    (+ NAX act_map epilogue on M5).
    gu = model.moe.switch_mlp.gate_up_proj
    dn = model.moe.switch_mlp.down_proj
    # Unsorted short call (per_row=top_k for gate_up, per_row=1 for down)
    x_short = mx.zeros((1, 2, 1, 1, hidden), dtype=dtype)
    idx_short = mx.zeros((1, 2, top_k), dtype=mx.uint32)
    h_short = gu(x_short, idx_short, sorted_indices=False)
    mx.eval(h_short)
    x_dn_short = mx.zeros((1, 2, top_k, 1, inter), dtype=dtype)
    y_dn_short = dn(x_dn_short, idx_short, sorted_indices=False)
    mx.eval(y_dn_short)
    # Sorted under-threshold call (per_row=1 for gate_up & down)
    x_mid = mx.zeros((64, 1, hidden), dtype=dtype)
    idx_mid = mx.zeros((64,), dtype=mx.uint32)
    mx.eval(gu(x_mid, idx_mid, sorted_indices=True))
    x_dn_mid = mx.zeros((64, 1, inter), dtype=dtype)
    mx.eval(dn(x_dn_mid, idx_mid, sorted_indices=True))
    # Sorted prefill above threshold (tile_gather on M3, NAX plain + act_map on M5)
    n_sorted = 4 * experts
    x_sorted = mx.zeros((n_sorted, 1, hidden), dtype=dtype)
    idx_sorted = mx.zeros((n_sorted,), dtype=mx.uint32)
    mx.eval(gu(x_sorted, idx_sorted, sorted_indices=True))
    x_dn_sorted = mx.zeros((n_sorted, 1, inter), dtype=dtype)
    mx.eval(dn(x_dn_sorted, idx_sorted, sorted_indices=True))
    if linear._nax_host():
        tok = mx.zeros((n_sorted // top_k, 1, hidden), dtype=dtype)
        row_map = mx.zeros((n_sorted,), dtype=mx.uint32)
        y_epi = gu._sorted_swiglu(x_sorted, idx_sorted, None, (tok, row_map))
        assert y_epi is not None
        mx.eval(y_epi)

    # 3. routed_decode (1-row) and routed_verify_window (2,3,4,6,8,9,12,16).
    for rows in (1, 2, 3, 4, 6, 8, 9, 12, 16):
        x_w = mx.zeros((1, rows, hidden), dtype=dtype)
        y_w = qwen35_moe_routed_decode.routed_verify_window(model.moe, x_w)
        assert y_w is not None
        mx.eval(y_w)

    # 4. moe_verify_gather (rows 2..8 with uint32 indices).
    for rows in (2, 3, 4, 8):
        x_vg = mx.zeros((1, rows, hidden), dtype=dtype)
        idx_vg = mx.zeros((1, rows, top_k), dtype=mx.uint32)
        y_vg = moe_verify_gather.fused_switch(model.moe.switch_mlp, x_vg, idx_vg)
        assert y_vg is not None
        mx.eval(y_vg)

    assert kernels.proof_count() == before, (
        f"expected 0 new proofs during served requests after load, "
        f"got {kernels.proof_count() - before}"
    )

    # 5. Verify bitwise-identical outputs against unwarmed fresh Proof path.
    mx.random.seed(99)
    x_rand = mx.random.normal((1, 4, hidden), dtype=dtype)
    y_warm_lin = model.proj_aligned(x_rand)
    y_warm_win = qwen35_moe_routed_decode.routed_verify_window(model.moe, x_rand)
    idx_rand = mx.zeros((1, 4, top_k), dtype=mx.uint32)
    y_warm_vg = moe_verify_gather.fused_switch(model.moe.switch_mlp, x_rand, idx_rand)
    mx.eval(y_warm_lin, y_warm_win, y_warm_vg)

    # Clear all proofs and cached plans, run cold without load warming, compare bits.
    monkeypatch.setattr(kernels, "_PROOFS", {})
    model.proj_aligned.__dict__.pop("_omlx_format_call", None)
    model.proj_aligned.__dict__.pop("_fast", None)
    model.moe.__dict__.pop("_omlx_routed_decode_plan", None)
    model.moe.switch_mlp.__dict__.pop("_omlx_verify_gather_plan", None)

    y_cold_lin = model.proj_aligned(x_rand)
    y_cold_win = qwen35_moe_routed_decode.routed_verify_window(model.moe, x_rand)
    y_cold_vg = moe_verify_gather.fused_switch(model.moe.switch_mlp, x_rand, idx_rand)
    mx.eval(y_cold_lin, y_cold_win, y_cold_vg)
    assert kernels.proof_count() > before, "unwarmed path should have compiled proofs"
    assert mx.array_equal(y_warm_lin.view(mx.uint16), y_cold_lin.view(mx.uint16)).item()
    assert mx.array_equal(y_warm_win.view(mx.uint16), y_cold_win.view(mx.uint16)).item()
    assert mx.array_equal(y_warm_vg.view(mx.uint16), y_cold_vg.view(mx.uint16)).item()


def test_proof_warm_fails_closed_on_none_launch(monkeypatch):
    """HIGH 1: Proof.warm must fail closed if the warm launch returns None or empty."""
    monkeypatch.setattr(kernels, "_PROOFS", {})
    proof = kernels.proof("test_warm_none", "k")

    for bad_result in (None, (), [None]):
        with pytest.raises(RuntimeError, match="returned no arrays"):
            proof.warm(
                "m",
                lambda stream, res=bad_result: (
                    (mx.zeros((1,), stream=stream),),
                    lambda _x, r=res: r,
                ),
            )
        assert "m" not in proof.ran


def test_proof_mark_warm_respects_failed_and_warm_is_idempotent(monkeypatch):
    """Surviving mutants 1 & 2: failed proofs stay unwarmed; warm skips already-ran keys."""
    monkeypatch.setattr(kernels, "_PROOFS", {})
    failed_proof = kernels.proof("test_mark_failed", "k")
    failed_proof.failed = True
    failed_proof._mark_warm("m")
    assert "m" not in failed_proof.ran

    proof = kernels.proof("test_warm_idempotent", "k")
    builds = []

    def build(stream):
        builds.append(stream)
        return (mx.zeros((1,), stream=stream),), lambda x: x + 1

    proof.warm("m", build)
    proof.warm("m", build)
    assert len(builds) == 1
    assert "m" in proof.ran


def test_warm_moe_model_joins_futures_on_plan_error(monkeypatch):
    """HIGH 2: warm_moe_model must join already-submitted futures if later planning raises."""
    fmt = affine.AffineFormat()
    dtype = mx.bfloat16

    class _Block(nn.Module):
        def __init__(self, seed):
            super().__init__()
            self.top_k = 2
            self.num_experts = 4
            self.switch_mlp = nn.Module()
            self.switch_mlp.gate_up_proj, _ = _quantized_format_switch(
                fmt, 4, 128, 64, 4, 64, dtype, seed=seed
            )
            self.switch_mlp.down_proj, _ = _quantized_format_switch(
                fmt, 4, 64, 64, 4, 64, dtype, seed=seed + 1
            )

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.b0 = _Block(seed=1)
            self.b1 = _Block(seed=10)

    with testing.verified("qwen4_exp"):
        model = _Model()

    awaited = []

    class _TrackedFuture:
        def result(self):
            awaited.append(1)

    calls = [0]

    def fake_warm_switch(layer, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            return (_TrackedFuture(),)
        raise ValueError("plan boom")

    monkeypatch.setattr(linear, "warm_existing_switch", fake_warm_switch)
    with pytest.raises(ValueError, match="plan boom"):
        qwen35_moe_routed_decode.warm_moe_model(model, mtp_enabled=False)
    assert len(awaited) == 1


def test_warm_moe_model_skips_non_standard_modules():
    """MEDIUM 3: non-standard MoE attributes and non-finite limits do not crash load."""
    fmt = affine.AffineFormat()
    dtype = mx.bfloat16

    class _BadSwitch(nn.Module):
        def __init__(self):
            super().__init__()
            self.top_k = 2
            self.num_experts = 4
            self.switch_mlp = object()

    class _BadTopK(nn.Module):
        def __init__(self):
            super().__init__()
            self.top_k = 0
            self.num_experts = 4
            self.switch_mlp = nn.Module()

    class _BoolTopK(nn.Module):
        def __init__(self):
            super().__init__()
            self.top_k = True
            self.num_experts = 4
            self.switch_mlp = nn.Module()

    class _EmptySwitchMlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.top_k = 2
            self.num_experts = 4
            self.switch_mlp = nn.Module()
            self.switch_mlp.gate_up_proj = nn.Module()
            self.switch_mlp.down_proj = nn.Module()
            self.gate = nn.Linear(64, 4, bias=False)
            self.gate.weight = mx.zeros((4, 64), dtype=dtype)

    class _InfAct:
        limit = float("inf")

    class _InfLimit(nn.Module):
        def __init__(self):
            super().__init__()
            self.top_k = 2
            self.num_experts = 4
            self.switch_mlp = nn.Module()
            self.switch_mlp.activation = _InfAct()
            self.switch_mlp.gate_up_proj, _ = _quantized_format_switch(
                fmt, 4, 128, 64, 4, 64, dtype, seed=50
            )
            self.switch_mlp.down_proj, _ = _quantized_format_switch(
                fmt, 4, 64, 64, 4, 64, dtype, seed=51
            )

    class _Container(nn.Module):
        def __init__(self):
            super().__init__()
            self.m0 = _BadSwitch()
            self.m1 = _BadTopK()
            self.m2 = _BoolTopK()
            self.m3 = _EmptySwitchMlp()
            self.m4 = _InfLimit()

    with testing.verified("qwen4_exp"):
        model = _Container()
        model.eval()
        qwen35_moe_routed_decode.warm_moe_model(model, mtp_enabled=False)


def test_fp16_recast_warms_dense_format_linear(monkeypatch, tmp_path):
    """MEDIUM 4: activation_fp16 recast warms dense FormatLinear for float16."""
    if not mx.metal.is_available():
        pytest.skip("requires Metal")

    from omlx import formats
    from omlx.utils.model_loading import cast_bf16_params_to_fp16

    formats.load()
    _write_qwen_config(tmp_path, dtype="bfloat16")
    monkeypatch.setattr(kernels, "_PROOFS", {})
    monkeypatch.setattr(kernels, "_RAW_KERNELS", {})
    fmt = affine.AffineFormat()

    class _DenseModel(nn.Module):
        def __init__(self, checkpoint):
            super().__init__()
            self.proj, lay = _quantized_format_linear(
                fmt, 256, 256, 8, 64, mx.bfloat16, seed=70
            )
            self.proj._stock = True
            tensors = {n: self.proj[n] for n in self.proj._tensors}
            format_load._warm_layer(self.proj, fmt, lay, 256, tensors, checkpoint)

    with format_load.loading(tmp_path) as checkpoint:
        checkpoint.verified = True
        checkpoint.verified_modes = frozenset({"affine"})
        model = _DenseModel(checkpoint)
        model.eval()

    recast = cast_bf16_params_to_fp16(model)
    assert recast > 0
    assert getattr(model, "_omlx_fp16_recast", False) is True

    qwen35_moe_routed_decode.warm_moe_model(model, mtp_enabled=False)
    before = kernels.proof_count()
    for rows in (1, 2, 4, 16, 32):
        y = model.proj(mx.zeros((1, rows, 256), dtype=mx.float16))
        mx.eval(y)
    assert kernels.proof_count() == before


def test_tile_dense_unbounded_wide_tail(monkeypatch):
    """LOW 6: unbounded wide tail past bucket_lo=32 warms both PARTIAL_M modes."""
    from omlx.formats import tile_dense

    fmt = affine.AffineFormat()
    with testing.verified("qwen4_exp"):
        layer, _ = _quantized_format_linear(fmt, 256, 256, 4, 64, mx.bfloat16, seed=80)

    real_plan = tile_dense.build_plan(layer, tensor_units=False)
    assert real_plan is not None
    steel = real_plan.schedule[-1][1]
    wide = tile_dense.DenseTilePolicy(backend="wide")
    custom_plan = tile_dense.DensePlan(
        face=real_plan.face,
        operands=real_plan.operands,
        proof=real_plan.proof,
        said=real_plan.said,
        k=real_plan.k,
        n=real_plan.n,
        bits=real_plan.bits,
        group_size=real_plan.group_size,
        schedule=((32, steel), (None, wide)),
        schedules={mx.bfloat16: ((32, steel), (None, wide))},
        row_limit=real_plan.row_limit,
        binding=real_plan.binding,
    )
    monkeypatch.setattr(tile_dense, "build_plan", lambda _l, **_kw: custom_plan)

    _, proofs = tile_dense.warm_specs(layer, (mx.bfloat16,), tensor_units=False)
    wide_metals = {
        metal
        for entries in proofs.values()
        for _proof, metal in entries
        if isinstance(metal, tuple) and metal[0] == "wide"
    }
    assert ("wide", mx.bfloat16, 256, 256, 5, False) in wide_metals
    assert ("wide", mx.bfloat16, 256, 256, 5, True) in wide_metals


def test_moe_verify_gather_warm_tasks_holds_no_arrays():
    """LOW 7: moe_verify_gather.warm_tasks closures must not retain mx.array weights."""
    import gc
    import weakref

    from mlx_vlm.models.switch_layers import SwiGLU
    from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

    fmt = affine.AffineFormat()
    dtype = mx.bfloat16

    def make_tasks():
        sw = VLMSwitchGLU.__new__(VLMSwitchGLU)
        nn.Module.__init__(sw)
        sw.activation = SwiGLU()
        with testing.verified("qwen4_exp"):
            sw.gate_up_proj, _ = _quantized_format_switch(
                fmt, 128, 256, 256, 4, 64, dtype, seed=90
            )
            sw.down_proj, _ = _quantized_format_switch(
                fmt, 128, 256, 128, 4, 64, dtype, seed=91
            )
        ref = weakref.ref(sw.gate_up_proj["weight"])
        tasks = moe_verify_gather.warm_tasks(sw, 256, 10, dtype)
        assert tasks
        return tasks, ref

    tasks, ref = make_tasks()
    gc.collect()
    assert ref() is None, "warm_tasks closure retained live weight mx.array"
    assert len(tasks) > 0


def test_warm_moe_model_populates_cached_plan(monkeypatch):
    """Optimization 1: warm_moe_model caches routed_decode plan so token 0 avoids 288 syncs."""
    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP
    from mlx_vlm.models.switch_layers import SwiGLU
    from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

    fmt = affine.AffineFormat()
    dtype = mx.bfloat16
    hidden, inter, experts, top_k = 512, 256, 128, 10

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = experts
            self.top_k = top_k
            self.gate = nn.Linear(hidden, experts, bias=False)
            self.gate.weight = mx.zeros((experts, hidden), dtype=dtype)
            sw = VLMSwitchGLU.__new__(VLMSwitchGLU)
            nn.Module.__init__(sw)
            sw.activation = SwiGLU()
            sw.gate_up_proj, _ = _quantized_format_switch(
                fmt, experts, 2 * inter, hidden, 4, 64, dtype, seed=101
            )
            sw.down_proj, _ = _quantized_format_switch(
                fmt, experts, hidden, inter, 4, 64, dtype, seed=102
            )
            self.switch_mlp = sw
            sh = Qwen3_5MLP.__new__(Qwen3_5MLP)
            nn.Module.__init__(sh)
            sh.gate_proj, _ = _quantized_format_linear(
                fmt, inter, hidden, 8, 64, dtype, seed=103
            )
            sh.up_proj, _ = _quantized_format_linear(
                fmt, inter, hidden, 8, 64, dtype, seed=104
            )
            sh.down_proj, _ = _quantized_format_linear(
                fmt, hidden, inter, 8, 64, dtype, seed=105
            )
            self.shared_expert = sh
            self.shared_expert_gate, _ = _quantized_format_linear(
                fmt, 1, hidden, 8, 64, dtype, seed=106
            )

    with testing.verified("qwen4_exp"):
        block = _Block()
        block.eval()

    monkeypatch.setattr(kernels, "warm", lambda _inst: ())
    monkeypatch.setattr(kernels, "warm_proofs", lambda _tasks: ())
    qwen35_moe_routed_decode.warm_moe_model(block, mtp_enabled=False)
    assert "_omlx_routed_decode_plan" in block.__dict__
    assert qwen35_moe_routed_decode._cached_plan(block, dtype) is not None


def test_surviving_mutants_config_experts_stackable_and_mtp_rows(monkeypatch, tmp_path):
    """Surviving mutants 3, 4, 5, 6, 7: config keys, experts dedup, stand-in shrink, MTP rows."""
    # Mutant 3: top-level torch_dtype and nested text_config.dtype both parsed.
    d1 = tmp_path / "top_torch_dtype"
    d1.mkdir()
    (d1 / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "torch_dtype": "bfloat16"})
    )
    with format_load.loading(d1) as ckpt1:
        assert ckpt1.warm_dtypes == (mx.bfloat16,)

    d2 = tmp_path / "nested_dtype"
    d2.mkdir()
    (d2 / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "text_config": {"dtype": "float16"}})
    )
    with format_load.loading(d2) as ckpt2:
        assert ckpt2.warm_dtypes == (mx.float16,)

    # Mutant 4: _warm_layer deduplication key includes num_experts for switch layers.
    fmt = affine.AffineFormat()
    seen_experts = []
    monkeypatch.setattr(
        format_load,
        "warm_layer",
        lambda cls, _fmt, _b, _g, _k, tensors, _lay, **_kw: seen_experts.append(
            tensors[_fmt.codes_name].shape[0]
        ),
    )
    with format_load.loading(d1) as ckpt:
        ckpt.verified = True
        ckpt.verified_modes = frozenset({"affine"})
        sw4, lay4 = _quantized_format_switch(fmt, 4, 64, 64, 4, 64, mx.bfloat16, seed=1)
        sw8, lay8 = _quantized_format_switch(fmt, 8, 64, 64, 4, 64, mx.bfloat16, seed=2)
        format_load._warm_layer(
            sw4, fmt, lay4, 64, {n: sw4[n] for n in sw4._tensors}, ckpt
        )
        format_load._warm_layer(
            sw8, fmt, lay8, 64, {n: sw8[n] for n in sw8._tensors}, ckpt
        )
    assert seen_experts == [4, 8]

    # Mutant 5: large-N affine stand-in shrinks rows below N (stackable=True).
    with testing.verified("qwen4_exp"):
        big_layer, _ = _quantized_format_linear(
            fmt, 2048, 256, 4, 64, mx.bfloat16, seed=3
        )
    instances: dict = {}
    proofs: dict = {}
    linear._plan_dense_warm(
        big_layer, fmt, 4, 64, 256, (mx.bfloat16,), instances, proofs
    )
    assert instances
    for recipe in instances.values():
        stand_in = recipe()
        # weight is input index 1; shrunk row count must be strictly < 2048.
        assert stand_in.input_shapes[1][0] < 2048

    # Mutant 6: mtp_enabled=False warms only rows (2,3,4), True warms (2,3,4,6,8,9,12,16).
    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP
    from mlx_vlm.models.switch_layers import SwiGLU
    from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

    class _MtpBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = 128
            self.top_k = 10
            self.gate = nn.Linear(512, 128, bias=False)
            self.gate.weight = mx.zeros((128, 512), dtype=mx.bfloat16)
            sw = VLMSwitchGLU.__new__(VLMSwitchGLU)
            nn.Module.__init__(sw)
            sw.activation = SwiGLU()
            sw.gate_up_proj, _ = _quantized_format_switch(
                fmt, 128, 512, 512, 4, 64, mx.bfloat16, seed=111
            )
            sw.down_proj, _ = _quantized_format_switch(
                fmt, 128, 512, 256, 4, 64, mx.bfloat16, seed=112
            )
            self.switch_mlp = sw
            sh = Qwen3_5MLP.__new__(Qwen3_5MLP)
            nn.Module.__init__(sh)
            sh.gate_proj, _ = _quantized_format_linear(
                fmt, 256, 512, 8, 64, mx.bfloat16, seed=113
            )
            sh.up_proj, _ = _quantized_format_linear(
                fmt, 256, 512, 8, 64, mx.bfloat16, seed=114
            )
            sh.down_proj, _ = _quantized_format_linear(
                fmt, 512, 256, 8, 64, mx.bfloat16, seed=115
            )
            self.shared_expert = sh
            self.shared_expert_gate, _ = _quantized_format_linear(
                fmt, 1, 512, 8, 64, mx.bfloat16, seed=116
            )

    with testing.verified("qwen4_exp"):
        block = _MtpBlock()
        block.eval()

    recorded = []
    monkeypatch.setattr(kernels, "warm", lambda _inst: ())
    monkeypatch.setattr(
        kernels,
        "warm_proofs",
        lambda tasks: recorded.extend((p, m) for p, m, _ in tasks) or (),
    )
    qwen35_moe_routed_decode.warm_moe_model(block, mtp_enabled=False)
    window_proof = qwen35_moe_routed_decode._cached_plan(block, mx.bfloat16).window
    off_rows = {m[0] for p, m in recorded if p is window_proof}
    assert off_rows == {2, 3, 4}

    recorded.clear()
    qwen35_moe_routed_decode.warm_moe_model(block, mtp_enabled=True)
    on_rows = {m[0] for p, m in recorded if p is window_proof}
    assert on_rows == {2, 3, 4, 6, 8, 9, 12, 16}

    # Mutant 7: engine load call sites forward model_settings.mtp_enabled (not hardcoded False).
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel in ("omlx/engine/batched.py", "omlx/engine/vlm.py"):
        src = (root / rel).read_text()
        assert 'getattr(self._model_settings, "mtp_enabled", False)' in src
        assert "warm_moe_model(" in src and "mtp_enabled=mtp_warm" in src


def test_first_served_request_does_no_keep_build_plan_compile_or_register_fast(
    monkeypatch, tmp_path
):
    """First served request runs 0 _keep, build_plan, mx.compile, _register_fast."""
    if not mx.metal.is_available():
        pytest.skip("requires Metal")

    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP
    from mlx_vlm.models.switch_layers import SwiGLU
    from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

    from omlx.formats import tile_dense

    _write_qwen_config(tmp_path, dtype="bfloat16")
    monkeypatch.setattr(kernels, "_PROOFS", {})
    monkeypatch.setattr(kernels, "_RAW_KERNELS", {})

    fmt = affine.AffineFormat()
    dtype = mx.bfloat16
    hidden, inter, experts, top_k = 512, 256, 128, 8
    prefill_len = 64

    class _ServedModel(nn.Module):
        def __init__(self, checkpoint):
            super().__init__()
            # Two sibling layers of identical geometry to prove shared compiled
            # dense graph + per-layer fast dispatch registration at load.
            self.proj_a, lay_a = _quantized_format_linear(
                fmt, 256, hidden, 8, 64, dtype, seed=201
            )
            self.proj_b, lay_b = _quantized_format_linear(
                fmt, 256, hidden, 8, 64, dtype, seed=202
            )
            self.proj_b._qwen_row_exact = qwen35_verify_qmm.is_row_exact_armed
            self.proj_n1, lay_n1 = _quantized_format_linear(
                fmt, 1, hidden, 8, 64, dtype, seed=203
            )
            self.moe = nn.Module()
            self.moe.num_experts = experts
            self.moe.top_k = top_k
            self.moe.gate = nn.Linear(hidden, experts, bias=False)
            self.moe.gate.weight = mx.random.normal(
                (experts, hidden), dtype=dtype
            ) * mx.array(0.05, dtype=dtype)
            mx.eval(self.moe.gate.weight)

            sw = VLMSwitchGLU.__new__(VLMSwitchGLU)
            nn.Module.__init__(sw)
            sw.activation = SwiGLU()
            sw.gate_proj, lay_g = _quantized_format_switch(
                fmt, experts, inter, hidden, 4, 64, dtype, seed=210
            )
            sw.up_proj, lay_u = _quantized_format_switch(
                fmt, experts, inter, hidden, 4, 64, dtype, seed=211
            )
            sw.down_proj, lay_d = _quantized_format_switch(
                fmt, experts, hidden, inter, 4, 64, dtype, seed=212
            )
            self.moe.switch_mlp = sw

            sh = Qwen3_5MLP.__new__(Qwen3_5MLP)
            nn.Module.__init__(sh)
            sh.gate_proj, lay_sg = _quantized_format_linear(
                fmt, inter, hidden, 8, 64, dtype, seed=220
            )
            sh.up_proj, lay_su = _quantized_format_linear(
                fmt, inter, hidden, 8, 64, dtype, seed=221
            )
            sh.down_proj, lay_sd = _quantized_format_linear(
                fmt, hidden, inter, 8, 64, dtype, seed=222
            )
            self.moe.shared_expert = sh
            self.moe.shared_expert_gate = self.proj_n1

            for mod, lay in (
                (self.proj_a, lay_a),
                (self.proj_b, lay_b),
                (self.proj_n1, lay_n1),
                (sw.gate_proj, lay_g),
                (sw.up_proj, lay_u),
                (sw.down_proj, lay_d),
                (sh.gate_proj, lay_sg),
                (sh.up_proj, lay_su),
                (sh.down_proj, lay_sd),
            ):
                tensors = {n: mod[n] for n in mod._tensors}
                format_load._warm_layer(
                    mod, fmt, lay, mod.input_dims, tensors, checkpoint
                )

    with format_load.loading(tmp_path) as checkpoint:
        checkpoint.verified = True
        checkpoint.verified_modes = frozenset({"affine"})
        model = _ServedModel(checkpoint)
        model.eval()

    qwen35_moe_gate_up._fuse_one(model.moe.switch_mlp)
    qwen35_moe_routed_decode.warm_moe_model(
        model, mtp_enabled=True, prefill_lengths=(prefill_len, prefill_len + 1)
    )

    keep_calls: list[object] = []
    build_plan_calls: list[object] = []
    compile_calls: list[object] = []
    register_fast_calls: list[object] = []

    orig_keep = linear._FormatLayer._keep
    orig_build_plan = tile_dense.build_plan
    orig_compile = mx.compile
    orig_register_fast = linear.FormatLinear._register_fast

    def spy_keep(self):
        keep_calls.append(self)
        return orig_keep(self)

    def spy_build_plan(*args, **kwargs):
        build_plan_calls.append(args[0])
        return orig_build_plan(*args, **kwargs)

    def spy_compile(*args, **kwargs):
        compile_calls.append(args[0] if args else None)
        return orig_compile(*args, **kwargs)

    def spy_register_fast(self, *args, **kwargs):
        register_fast_calls.append((self, args[1] if len(args) > 1 else None))
        return orig_register_fast(self, *args, **kwargs)

    monkeypatch.setattr(linear._FormatLayer, "_keep", spy_keep)
    monkeypatch.setattr(tile_dense, "build_plan", spy_build_plan)
    monkeypatch.setattr(mx, "compile", spy_compile)
    monkeypatch.setattr(linear.FormatLinear, "_register_fast", spy_register_fast)

    before_proofs = kernels.proof_count()

    mx.random.seed(77)
    x_pre = mx.random.normal((1, prefill_len, hidden), dtype=dtype)
    x_pre2 = mx.random.normal((1, prefill_len + 1, hidden), dtype=dtype)
    x_dec = mx.random.normal((1, 1, hidden), dtype=dtype)
    x_b2 = mx.random.normal((2, 1, hidden), dtype=dtype)
    x_b4 = mx.random.normal((4, 1, hidden), dtype=dtype)
    x_ver = mx.random.normal((1, 4, hidden), dtype=dtype)
    n_sorted = prefill_len * top_k
    x_sw = mx.random.normal((n_sorted, 1, hidden), dtype=dtype)
    idx_sw = mx.sort(mx.random.randint(0, experts, (n_sorted,), dtype=mx.uint32))
    x_sw_dn = mx.random.normal((n_sorted, 1, inter), dtype=dtype)
    mx.eval(x_pre, x_pre2, x_dec, x_b2, x_b4, x_ver, x_sw, idx_sw, x_sw_dn)

    # First served prefill (L and L+1) + decode + batched decode (b2/b4 -> prefill)
    # + armed verify + switch + routed MoE.
    y_pre_a = model.proj_a(x_pre)
    y_pre_b = model.proj_b(x_pre)
    y_pre2_a = model.proj_a(x_pre2)
    y_pre_n1 = model.proj_n1(x_pre)
    y_dec_a = model.proj_a(x_dec)
    y_dec_b = model.proj_b(x_dec)
    y_b2_b = model.proj_b(x_b2)
    y_b4_b = model.proj_b(x_b4)
    y_pre_b_again = model.proj_b(x_pre)
    y_dec_n1 = model.proj_n1(x_dec)

    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=True)
    try:
        y_ver_a = model.proj_a(x_ver)
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False, row_exact=False)

    gu = model.moe.switch_mlp.gate_up_proj
    dn = model.moe.switch_mlp.down_proj
    y_sw_gu = gu(x_sw, idx_sw, sorted_indices=True)
    y_sw_dn = dn(x_sw_dn, idx_sw, sorted_indices=True)
    y_sw_epi = None
    if linear._nax_host():
        tok = mx.random.normal((prefill_len, 1, hidden), dtype=dtype)
        row_map = mx.arange(n_sorted, dtype=mx.uint32) // top_k
        mx.eval(tok, row_map)
        y_sw_epi = gu._sorted_swiglu(x_sw, idx_sw, None, (tok, row_map))
        assert y_sw_epi is not None

    y_rd = qwen35_moe_routed_decode.routed_verify_window(model.moe, x_dec)
    y_rb2 = qwen35_moe_routed_decode.routed_verify_window(model.moe, x_b2)
    y_rw = qwen35_moe_routed_decode.routed_verify_window(model.moe, x_ver)
    assert y_rd is not None and y_rb2 is not None and y_rw is not None

    to_eval = [
        y_pre_a,
        y_pre_b,
        y_pre2_a,
        y_pre_n1,
        y_dec_a,
        y_dec_b,
        y_b2_b,
        y_b4_b,
        y_pre_b_again,
        y_dec_n1,
        y_ver_a,
        y_sw_gu,
        y_sw_dn,
        y_rd,
        y_rb2,
        y_rw,
    ]
    if y_sw_epi is not None:
        to_eval.append(y_sw_epi)
    mx.eval(*to_eval)

    assert keep_calls == [], f"unexpected _keep calls: {len(keep_calls)}"
    assert (
        build_plan_calls == []
    ), f"unexpected build_plan calls: {len(build_plan_calls)}"
    assert compile_calls == [], f"unexpected mx.compile calls: {len(compile_calls)}"
    assert (
        register_fast_calls == []
    ), f"unexpected _register_fast calls: {len(register_fast_calls)}"
    assert kernels.proof_count() == before_proofs

    # Negative control + bitwise exact check: clear caches on proj_a and proj_b and
    # verify spies fire AND outputs are bitwise identical (uint16).
    monkeypatch.setattr(kernels, "_PROOFS", {})
    model.proj_a.__dict__.pop("_omlx_format_call", None)
    model.proj_a.__dict__.pop("_fast", None)
    model.proj_b.__dict__.pop("_omlx_format_call", None)
    model.proj_b.__dict__.pop("_fast", None)
    y_cold_pre = model.proj_a(x_pre)
    y_cold_dec = model.proj_a(x_dec)
    y_cold_b2 = model.proj_b(x_b2)
    y_cold_pre_b = model.proj_b(x_pre)
    mx.eval(y_cold_pre, y_cold_dec, y_cold_b2, y_cold_pre_b)
    assert len(keep_calls) >= 1
    assert len(build_plan_calls) >= 1
    assert len(compile_calls) >= 1
    assert len(register_fast_calls) >= 1
    assert mx.array_equal(y_pre_a.view(mx.uint16), y_cold_pre.view(mx.uint16)).item()
    assert mx.array_equal(y_dec_a.view(mx.uint16), y_cold_dec.view(mx.uint16)).item()
    assert mx.array_equal(y_b2_b.view(mx.uint16), y_cold_b2.view(mx.uint16)).item()
    assert mx.array_equal(
        y_pre_b_again.view(mx.uint16), y_cold_pre_b.view(mx.uint16)
    ).item()


def test_warm_engine_dispatch_populates_engine_thread_and_preserves_thread_safety(
    monkeypatch, tmp_path
):
    """warm_engine_dispatch warms the engine thread while keeping other threads safe."""
    if not mx.metal.is_available():
        pytest.skip("requires Metal")

    import concurrent.futures

    from omlx.formats import tile_dense

    _write_qwen_config(tmp_path, dtype="bfloat16")
    monkeypatch.setattr(kernels, "_PROOFS", {})
    monkeypatch.setattr(kernels, "_RAW_KERNELS", {})
    fmt = affine.AffineFormat()
    dtype = mx.bfloat16
    hidden = 512

    class _Model(nn.Module):
        def __init__(self, checkpoint):
            super().__init__()
            self.p0, lay0 = _quantized_format_linear(
                fmt, 256, hidden, 8, 64, dtype, seed=240
            )
            self.p1, lay1 = _quantized_format_linear(
                fmt, 256, hidden, 8, 64, dtype, seed=241
            )
            for mod, lay in ((self.p0, lay0), (self.p1, lay1)):
                tensors = {n: mod[n] for n in mod._tensors}
                format_load._warm_layer(
                    mod, fmt, lay, mod.input_dims, tensors, checkpoint
                )

    with format_load.loading(tmp_path) as checkpoint:
        checkpoint.verified = True
        checkpoint.verified_modes = frozenset({"affine"})
        model = _Model(checkpoint)
        model.eval()

    # Loader thread warms Metals + thread-independent plans only (warm_dispatch=False).
    qwen35_moe_routed_decode.warm_moe_model(
        model, mtp_enabled=False, warm_dispatch=False
    )
    assert "_omlx_format_call" in model.p0.__dict__
    assert model.p0.__dict__["_omlx_format_call"].dense is not linear._UNBUILT

    mx.random.seed(242)
    x_64 = mx.random.normal((1, 64, hidden), dtype=dtype)
    x_65 = mx.random.normal((1, 65, hidden), dtype=dtype)
    x_1 = mx.random.normal((1, 1, hidden), dtype=dtype)
    mx.eval(x_64, x_65, x_1)

    engine_stream = mx.new_thread_local_stream(mx.gpu)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as engine_exec:
        engine_exec.submit(
            qwen35_moe_routed_decode.warm_engine_dispatch,
            model,
            engine_stream,
            prefill_step_size=65,
            mtp_enabled=False,
        ).result()

        counts = {"keep": 0, "build_plan": 0, "compile": 0, "register_fast": 0}
        orig_keep = linear._FormatLayer._keep
        orig_build_plan = tile_dense.build_plan
        orig_compile = mx.compile
        orig_register_fast = linear.FormatLinear._register_fast

        def _wrap(name, fn):
            def wrapped(*args, **kwargs):
                counts[name] += 1
                return fn(*args, **kwargs)

            return wrapped

        monkeypatch.setattr(linear._FormatLayer, "_keep", _wrap("keep", orig_keep))
        monkeypatch.setattr(
            tile_dense, "build_plan", _wrap("build_plan", orig_build_plan)
        )
        monkeypatch.setattr(mx, "compile", _wrap("compile", orig_compile))
        monkeypatch.setattr(
            linear.FormatLinear,
            "_register_fast",
            _wrap("register_fast", orig_register_fast),
        )

        before = kernels.proof_count()

        def _serve():
            with mx.stream(engine_stream):
                y64 = model.p0(x_64)
                y65 = model.p1(x_65)
                y1 = model.p0(x_1)
                mx.eval(y64, y65, y1)
                return y64, y65, y1

        y64_t1, y65_t1, y1_t1 = engine_exec.submit(_serve).result()
        assert counts == {
            "keep": 0,
            "build_plan": 0,
            "compile": 0,
            "register_fast": 0,
        }
        assert kernels.proof_count() == before

    # A second independent thread must safely build its own thread-local graphs
    # and produce bitwise-identical outputs (protocol rule 14).
    other_stream = mx.new_thread_local_stream(mx.gpu)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as other_exec:

        def _serve_other():
            with mx.stream(other_stream):
                y64 = model.p0(x_64)
                y65 = model.p1(x_65)
                y1 = model.p0(x_1)
                mx.eval(y64, y65, y1)
                return y64, y65, y1

        y64_t2, y65_t2, y1_t2 = other_exec.submit(_serve_other).result()

    assert mx.array_equal(y64_t1.view(mx.uint16), y64_t2.view(mx.uint16)).item()
    assert mx.array_equal(y65_t1.view(mx.uint16), y65_t2.view(mx.uint16)).item()
    assert mx.array_equal(y1_t1.view(mx.uint16), y1_t2.view(mx.uint16)).item()


def test_prepare_layer_plans_fails_closed_on_placeholder_and_never_keeps_unproven_graph(
    monkeypatch,
):
    """Fail closed on placeholders; never insert unproven Metals into graphs."""
    from omlx.formats import tile_dense

    monkeypatch.setattr(kernels, "_PROOFS", {})
    fmt = affine.AffineFormat()
    dtype = mx.bfloat16
    layout = fmt.layout(256, 256, 8, 64)

    class _Holder(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

    with testing.verified("qwen4_exp"):
        placeholder_layer = FormatLinear(
            fmt,
            8,
            64,
            256,
            {
                name: mx.zeros(shape, dtype if dt is None else dt)
                for name, (shape, dt) in layout.items()
            },
            None,
            placeholders=True,
            _load_layout=layout,
        )
        unproven_layer, _ = _quantized_format_linear(
            fmt, 256, 256, 8, 64, dtype, seed=250
        )

    with pytest.raises(ValueError, match="placeholder"):
        linear.prepare_layer_plans(_Holder(placeholder_layer))

    # Without load-time Metal warming, Proof.ran is empty; warm_model_dispatch
    # must NOT insert unproven Metal graphs into proof.graphs or binding.kept.graphs.
    holder = _Holder(unproven_layer)
    linear.warm_model_dispatch(holder, prefill_lengths=(64,), mtp_enabled=False)
    kept = unproven_layer.__dict__["_omlx_format_call"]
    assert kept.plan is not None
    assert kept.plan.proof.graphs == {}
    steel_plan = tile_dense.build_plan(unproven_layer, tensor_units=False)
    assert steel_plan is not None
    steel_policy = tile_dense._default_steel_policy(128, 256, 256)
    forced_steel = tile_dense.DensePlan(
        face=steel_plan.face,
        operands=steel_plan.operands,
        proof=kernels.proof("test_unproven_steel", steel_plan.face),
        said=steel_plan.said,
        schedule=((None, steel_policy),),
        schedules={dtype: ((None, steel_policy),)},
        k=steel_plan.k,
        n=steel_plan.n,
        bits=steel_plan.bits,
        group_size=steel_plan.group_size,
        row_limit=steel_plan.row_limit,
        binding=kernels.proof("test_unproven_steel", steel_plan.face).bind(),
    )
    assert tile_dense.warm_graph(forced_steel, (1, 128, 256), dtype) is False
    assert forced_steel.proof.graphs == {}
    assert forced_steel.binding.kept.graphs == {}


def test_engine_load_wires_warm_engine_dispatch_with_fail_closed_cleanup():
    """Engine loaders run warm_engine_dispatch on the engine thread & fail closed."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel in ("omlx/engine/batched.py", "omlx/engine/vlm.py"):
        src = (root / rel).read_text()
        assert "warm_dispatch=False" in src
        assert "warm_engine_dispatch(" in src
        assert "engine_prefill_lengths(scheduler)" in src
        assert "self._engine.engine._mlx_executor" in src
        assert "self._engine.engine._mlx_stream" in src
        assert "await self.stop()" in src
        assert src.index("scheduler.refresh_ssd_layer_signature()") < src.index(
            "warm_engine_dispatch("
        )
    vlm_src = (root / "omlx/engine/vlm.py").read_text()
    assert vlm_src.index("apply_qwen35_q4_prefill_linear_patch(") < vlm_src.index(
        "warm_engine_dispatch("
    )

    class _FakeSched:
        def __init__(self, first, later):
            self._first = first
            self._later = later

        def _base_prefill_step_size(self, processed, _remaining):
            return self._first if processed == 0 else self._later

    assert qwen35_moe_routed_decode.engine_prefill_lengths(_FakeSched(2048, 2048)) == (
        2047,
        2048,
    )
    assert qwen35_moe_routed_decode.engine_prefill_lengths(_FakeSched(4096, 4096)) == (
        2047,
        2048,
        4095,
        4096,
    )
    assert qwen35_moe_routed_decode.engine_prefill_lengths(_FakeSched(2048, 4096)) == (
        2047,
        2048,
        4095,
        4096,
    )


def test_warm_existing_switch_skips_already_failed_tile_proof(monkeypatch):
    """pre-a1-c936f207 LOW 1: already-failed tile/epilogue proofs do not crash load."""
    monkeypatch.setattr(kernels, "_PROOFS", {})
    fmt = affine.AffineFormat()
    dtype = mx.bfloat16
    with testing.verified("qwen4_exp"):
        layer, _ = _quantized_format_switch(fmt, 4, 128, 64, 4, 64, dtype, seed=260)

    tile = linear._tile(layer)
    plan = linear._plan(layer)
    assert tile is not None and plan is not None
    tile.proof.fail()
    if linear._nax_host():
        linear._epilogue_proof(tile, plan, mapped=True).fail()

    futs = linear.warm_existing_switch(
        layer, dtypes=(dtype,), per_rows=(1,), epilogue_limit=None
    )
    for fut in futs or ():
        fut.result()


def test_c936_comment_review_mutants(monkeypatch):
    """Kill all 5 surviving mutants from pre-a1-c936f207 review."""
    fmt = affine.AffineFormat()
    dtype = mx.bfloat16

    # Mutant 1: primary exception is not masked when an already-submitted future raises.
    class _Block(nn.Module):
        def __init__(self, seed):
            super().__init__()
            self.top_k = 2
            self.num_experts = 4
            self.switch_mlp = nn.Module()
            self.switch_mlp.gate_up_proj, _ = _quantized_format_switch(
                fmt, 4, 128, 64, 4, 64, dtype, seed=seed
            )
            self.switch_mlp.down_proj, _ = _quantized_format_switch(
                fmt, 4, 64, 64, 4, 64, dtype, seed=seed + 1
            )

    class _TwoBlocks(nn.Module):
        def __init__(self):
            super().__init__()
            self.b0 = _Block(seed=270)
            self.b1 = _Block(seed=275)

    with testing.verified("qwen4_exp"):
        two = _TwoBlocks()

    failing_future = _Future(error=RuntimeError("secondary warm failure"))
    calls = [0]

    def fake_warm_switch(layer, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            return (failing_future,)
        raise ValueError("primary plan boom")

    monkeypatch.setattr(linear, "warm_existing_switch", fake_warm_switch)
    with pytest.raises(ValueError, match="primary plan boom"):
        qwen35_moe_routed_decode.warm_moe_model(two, mtp_enabled=False)
    assert failing_future.awaited == 1

    # Mutant 2 & 3: invalid num_experts (0 / True) skipped; bool epilogue limit ignored.
    seen_limits = []
    monkeypatch.setattr(
        linear,
        "warm_existing_switch",
        lambda layer, **kwargs: (
            seen_limits.append(kwargs.get("epilogue_limit", linear._NO_EPILOGUE)) or ()
        ),
    )
    monkeypatch.setattr(kernels, "warm", lambda _inst: ())
    monkeypatch.setattr(kernels, "warm_proofs", lambda _tasks: ())

    class _BoolLimitAct:
        limit = True

    class _EdgeContainer(nn.Module):
        def __init__(self):
            super().__init__()
            self.bad_e0 = nn.Module()
            self.bad_e0.top_k = 2
            self.bad_e0.num_experts = 0
            self.bad_e0.switch_mlp = nn.Module()
            self.bad_eb = nn.Module()
            self.bad_eb.top_k = 2
            self.bad_eb.num_experts = True
            self.bad_eb.switch_mlp = nn.Module()
            self.bool_lim = _Block(seed=280)
            self.bool_lim.switch_mlp.activation = _BoolLimitAct()

    from omlx.patches import m5_gather_qmm

    monkeypatch.setattr(
        m5_gather_qmm,
        "_swiglu_limit",
        lambda act: (
            True if isinstance(act, _BoolLimitAct) else m5_gather_qmm._UNSUPPORTED
        ),
    )
    with testing.verified("qwen4_exp"):
        edge = _EdgeContainer()
        edge.eval()
        qwen35_moe_routed_decode.warm_moe_model(edge, mtp_enabled=False)
    assert seen_limits and all(lim is linear._NO_EPILOGUE for lim in seen_limits)

    # Mutant 4: _nax_face_plans queries all 5 sample row counts.
    from omlx.patches import m5_gather_qmm_nax as nax

    queried_rows = []
    monkeypatch.setattr(
        nax,
        "face_plan",
        lambda r, e, k, n: queried_rows.append(r) or ("plan", r),
    )
    linear._nax_face_plans(8, 256, 256)
    assert queried_rows == [16, 36 * 8, 48 * 8, 96 * 8, 120 * 8]

    # Mutant 5: warm_existing_dense dedup key includes tensor_dtypes.
    planned_dtypes = []
    monkeypatch.setattr(
        linear,
        "_plan_dense_warm",
        lambda mod, *args: planned_dtypes.append(mod["scales"].dtype),
    )

    class _MixedDtypeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bf, _ = _quantized_format_linear(
                fmt, 256, 256, 8, 64, mx.bfloat16, seed=290
            )
            self.fp, _ = _quantized_format_linear(
                fmt, 256, 256, 8, 64, mx.float16, seed=291
            )

    with testing.verified("qwen4_exp"):
        mixed = _MixedDtypeModel()
    linear.warm_existing_dense(mixed, dtypes=(mx.float16,))
    assert set(planned_dtypes) == {mx.bfloat16, mx.float16}


def test_tile_dense_schedule_cache_matches_uncached_and_respects_subclasses(
    monkeypatch,
):
    """build_plan(_schedule_cache=...) matches uncached and skips subclasses."""
    from omlx.formats import tile_dense, tuning

    fmt = affine.AffineFormat()
    dtype = mx.bfloat16
    with testing.verified("qwen4_exp"):
        a, _ = _quantized_format_linear(fmt, 256, 512, 8, 64, dtype, seed=301)
        b, _ = _quantized_format_linear(fmt, 256, 512, 8, 64, dtype, seed=302)

    class _NoOrdinaryRows(FormatLinear):
        def _ordinary_serves(self, rows: int, *, chip: str | None = None) -> bool:
            return False

    c = _NoOrdinaryRows.__new__(_NoOrdinaryRows)
    with testing.verified("qwen4_exp"):
        c._init_tensors(
            fmt,
            8,
            64,
            512,
            {n: a[n] for n in a._tensors},
            None,
            layout=fmt.layout(512, 256, 8, 64),
        )

    a._keep()
    b._keep()
    c._keep()

    cache: dict = {}
    p_uncached = tile_dense.build_plan(a, tensor_units=False)
    p_a = tile_dense.build_plan(a, tensor_units=False, _schedule_cache=cache)

    lookups = []
    orig_lookup = tuning.lookup

    def spy_lookup(*args, **kwargs):
        lookups.append(args)
        return orig_lookup(*args, **kwargs)

    monkeypatch.setattr(tuning, "lookup", spy_lookup)
    p_b = tile_dense.build_plan(b, tensor_units=False, _schedule_cache=cache)
    assert lookups == [], "identical FormatLinear should reuse _schedule_cache"

    p_c = tile_dense.build_plan(c, tensor_units=False, _schedule_cache=cache)
    assert len(lookups) > 0, "subclass overriding _ordinary_serves must bypass cache"
    assert (
        p_uncached is not None
        and p_a is not None
        and p_b is not None
        and p_c is not None
    )
    assert p_a.schedules == p_uncached.schedules
    assert p_b.schedules == p_uncached.schedules
    assert p_c.schedules == tile_dense.build_plan(c, tensor_units=False).schedules
    assert len(cache) == len(p_a.schedules)


def test_warm_row_exact_qmv_zero_compiles_on_first_request():
    """F1: zero omlx_row_exact_qmv Metal libraries compile during request #0."""
    import subprocess
    import sys
    from pathlib import Path

    if not mx.metal.is_available():
        pytest.skip("requires Metal")

    repo_root = str(Path(__file__).resolve().parents[1])
    probe = r"""
import ctypes as C
import ctypes.util
import json
import re
import sys
import threading

import mlx.core as mx
import mlx.nn as nn

from omlx.formats import affine, linear, testing
from omlx.formats.linear import FormatLinear
from omlx.patches import (
    mlx_vlm_qwen4_exp_compat as compat,
    qwen35_gdn_prework,
    qwen35_moe_routed_decode,
    row_exact_qmv,
)

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen4_exp.language import (
    Qwen4ExpGatedDeltaNet,
    Qwen4ExpRMSNormGated,
    _fused_tile,
)

mtp_enabled = bool(int(sys.argv[1]))

# Swizzle MTLDevice newLibraryWithSource:options:error: to count every real
# Metal library compilation of omlx_row_exact_qmv* by phase.
objc = C.CDLL(ctypes.util.find_library("objc"))
metal = C.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
ptr = C.c_void_p
metal.MTLCreateSystemDefaultDevice.restype = ptr
objc.object_getClass.argtypes = [ptr]
objc.object_getClass.restype = ptr
objc.sel_registerName.argtypes = [C.c_char_p]
objc.sel_registerName.restype = ptr
objc.class_getInstanceMethod.argtypes = [ptr, ptr]
objc.class_getInstanceMethod.restype = ptr
objc.method_getImplementation.argtypes = [ptr]
objc.method_getImplementation.restype = ptr
objc.method_setImplementation.argtypes = [ptr, ptr]
objc.method_setImplementation.restype = ptr

msg_str = C.CFUNCTYPE(C.c_char_p, ptr, ptr)(("objc_msgSend", objc))
sel_utf8 = objc.sel_registerName(b"UTF8String")

dev = metal.MTLCreateSystemDefaultDevice()
cls = objc.object_getClass(dev)
sel = objc.sel_registerName(b"newLibraryWithSource:options:error:")
method = objc.class_getInstanceMethod(cls, sel)
sig = C.CFUNCTYPE(ptr, ptr, ptr, ptr, ptr, ptr)
old_impl = sig(objc.method_getImplementation(method))

lock = threading.Lock()
phase = ["init"]
recorded = {"init": [], "load": [], "request0": []}


@sig
def hooked(self_p, cmd_p, src_p, opts_p, err_p):
    res = old_impl(self_p, cmd_p, src_p, opts_p, err_p)
    if src_p:
        raw = msg_str(src_p, sel_utf8)
        if raw:
            text = raw.decode("utf-8", errors="ignore")
            m = re.search(r"omlx_row_exact_qmv[A-Za-z0-9_]*", text)
            if m:
                with lock:
                    recorded[phase[0]].append(m.group(0))
    return res


objc.method_setImplementation(method, C.cast(hooked, ptr))

fmt = affine.AffineFormat()
dtype = mx.bfloat16


def make_fl(n, k, seed):
    mx.random.seed(seed)
    w_fp = (mx.random.normal((n, k)) * 0.02).astype(dtype)
    w, s, b = mx.quantize(w_fp, group_size=64, bits=8)
    mx.eval(w, s, b)
    layout = fmt.layout(k, n, 8, 64)
    return FormatLinear(
        fmt,
        8,
        64,
        k,
        {"weight": w, "scales": s, "biases": b},
        None,
        placeholders=False,
        _load_layout=layout,
    )


class _FakeIndexer(nn.Module):
    def __init__(self):
        super().__init__()
        self.index_qk_proj = make_fl(640, 2560, 15)


class _FakeAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = make_fl(12288, 2560, 11)
        self.k_proj = make_fl(512, 2560, 12)
        self.v_proj = make_fl(512, 2560, 13)
        self.o_proj = make_fl(2560, 6144, 14)
        self.indexer = _FakeIndexer()


def make_gdn():
    hk, hv, dk, dv = 16, 48, 128, 128
    c_dim = 2 * hk * dk + hv * dv
    mod = Qwen4ExpGatedDeltaNet.__new__(Qwen4ExpGatedDeltaNet)
    nn.Module.__init__(mod)
    mod.num_k_heads, mod.num_v_heads = hk, hv
    mod.head_k_dim, mod.head_v_dim = dk, dv
    mod.conv_kernel_size = 4
    mod.conv1d = nn.Conv1d(c_dim, c_dim, 4, groups=c_dim, bias=False)
    mod.conv1d.weight = mx.zeros((c_dim, 4, 1), dtype=dtype)
    mod.norm = Qwen4ExpRMSNormGated(dv, eps=1e-6, activation="sigmoid")
    mod.norm.weight = mx.ones((dv,), dtype=dtype)
    mod.A_log = mx.zeros((hv,), dtype=dtype)
    mod.dt_bias = mx.zeros((hv,), dtype=dtype)
    mod.in_proj_qkv = make_fl(c_dim, 2560, 21)
    mod.in_proj_z = make_fl(hv * dv, 2560, 22)
    mod.in_proj_b = make_fl(hv, 2560, 23)
    mod.in_proj_a = make_fl(hv, 2560, 24)
    mod.out_proj = make_fl(2560, hv * dv, 25)
    mod.eval()
    mx.eval(mod.parameters())
    return mod


class _MiniQwen4(nn.Module):
    def __init__(self):
        super().__init__()
        self.gdn = make_gdn()
        self.attn = _FakeAttn()


with testing.verified("qwen4_exp"):
    model = _MiniQwen4()
    model.eval()
    # Warm base FormatLinear 1-row Metals as format_load does.
    for fut in linear.warm_existing_dense(model, dtypes=(dtype,)) or ():
        fut.result()

# Phase 1: load-time warm.
phase[0] = "load"
qwen35_moe_routed_decode.warm_moe_model(
    model, mtp_enabled=mtp_enabled, warm_dispatch=False
)
stream = mx.default_stream(mx.gpu)
qwen35_moe_routed_decode.warm_engine_dispatch(
    model, stream, prefill_lengths=(), mtp_enabled=mtp_enabled
)

# Phase 2: first served request.
phase[0] = "request0"
mx.random.seed(99)
x1 = (mx.random.normal((1, 1, 2560)) * 0.2).astype(dtype)
xo1 = (mx.random.normal((1, 1, 6144)) * 0.2).astype(dtype)
mx.eval(x1, xo1)

# 1-row GDN in/out + 1-row Attention group4/group1.
dec_plan = qwen35_gdn_prework._qwen4_decode_plan(model.gdn)
assert dec_plan is not None
y_gdn_in = dec_plan[-2](x1)
y_gdn_out = dec_plan[-1](xo1)
g4_plan = row_exact_qmv._group_plan(
    (
        model.attn.q_proj,
        model.attn.k_proj,
        model.attn.v_proj,
        model.attn.indexer.index_qk_proj,
    ),
    x1,
    1,
    _fused_tile(1),
)
assert g4_plan is not None
y_g4 = row_exact_qmv._launch_group(g4_plan, x1, 1, x1.shape)
y_g1 = row_exact_qmv.quantized_linears_tiled(
    (model.attn.o_proj,), xo1, *_fused_tile(1)
)
assert y_g1 is not None
mx.eval(y_gdn_in, y_gdn_out, *y_g4, *y_g1)

bitwise_ok = True
# Verify 1-row bitwise equality vs direct FormatLinear calls.
ref_gdn_out = model.gdn.out_proj(xo1)
ref_o = model.attn.o_proj(xo1)
ref_q = model.attn.q_proj(x1)
mx.eval(ref_gdn_out, ref_o, ref_q)
bitwise_ok &= bool(
    mx.array_equal(y_gdn_out.view(mx.uint16), ref_gdn_out.view(mx.uint16)).item()
)
bitwise_ok &= bool(
    mx.array_equal(y_g1[0].view(mx.uint16), ref_o.view(mx.uint16)).item()
)
bitwise_ok &= bool(
    mx.array_equal(y_g4[0].view(mx.uint16), ref_q.view(mx.uint16)).item()
)

if mtp_enabled:
    ver_plan = qwen35_gdn_prework._qwen4_verify_plan(model.gdn)
    assert ver_plan is not None
    for rows in (2, 3, 4):
        xr = (mx.random.normal((1, rows, 2560)) * 0.2).astype(dtype)
        xor = (mx.random.normal((1, rows, 6144)) * 0.2).astype(dtype)
        mx.eval(xr, xor)
        yv_in = ver_plan[0](xr)
        yv_out = ver_plan[1](xor)
        p4 = row_exact_qmv._group_plan(
            (
                model.attn.q_proj,
                model.attn.k_proj,
                model.attn.v_proj,
                model.attn.indexer.index_qk_proj,
            ),
            xr,
            rows,
            _fused_tile(rows),
        )
        assert p4 is not None
        yg4 = row_exact_qmv._launch_group(p4, xr, rows, xr.shape)
        yg1 = row_exact_qmv.quantized_linears_tiled(
            (model.attn.o_proj,), xor, *_fused_tile(rows)
        )
        assert yg1 is not None
        yg3 = row_exact_qmv.quantized_linears(
            (model.attn.q_proj, model.attn.k_proj, model.attn.v_proj), xr
        )
        mx.eval(yv_in, yv_out, *yg4, *yg1, *yg3)
        # Bitwise check row 0 against serial 1-row FormatLinear.
        ref_r0 = model.attn.o_proj(xor[:, :1, :])
        mx.eval(ref_r0)
        bitwise_ok &= bool(
            mx.array_equal(
                yg1[0][:, :1, :].view(mx.uint16), ref_r0.view(mx.uint16)
            ).item()
        )
        bitwise_ok &= bool(
            mx.array_equal(
                yg3[0].view(mx.uint16), yg4[0].view(mx.uint16)
            ).item()
        )

print(json.dumps({"recorded": recorded, "bitwise_ok": bitwise_ok}))
"""
    for mtp_flag in (0, 1):
        res = subprocess.run(
            [sys.executable, "-c", probe, str(mtp_flag)],
            capture_output=True,
            text=True,
            check=False,
            env={**__import__("os").environ, "PYTHONPATH": repo_root},
        )
        assert res.returncode == 0, f"mtp={mtp_flag} probe failed:\n{res.stderr}"
        data = json.loads(res.stdout.strip().splitlines()[-1])
        assert data["bitwise_ok"] is True
        req0 = data["recorded"]["request0"]
        load = data["recorded"]["load"]
        assert req0 == [], (
            f"mtp={mtp_flag}: compiled omlx_row_exact_qmv libraries during "
            f"request #0: {req0}"
        )
        assert len(load) >= 4, f"mtp={mtp_flag}: expected >=4 load compiles, got {load}"
        if mtp_flag == 0:
            assert not any(
                "_u__" in name or "group3" in name for name in load
            ), f"mtp=0 should not compile verify libraries: {load}"
        else:
            assert any("group3" in name for name in load), f"missing group3 in {load}"
            assert any("_u__" in name for name in load), f"missing _u verify in {load}"


def test_warm_row_exact_qmv_fails_closed_and_verifies_real_weights(monkeypatch):
    """F1 fails closed on warm error and never poisons _UNROLLED_VERDICTS."""
    from omlx.patches import row_exact_qmv

    if not mx.metal.is_available():
        pytest.skip("requires Metal")

    fmt = affine.AffineFormat()
    dtype = mx.bfloat16

    class _FakeIndexer(nn.Module):
        def __init__(self):
            super().__init__()
            self.index_qk_proj, _ = _quantized_format_linear(
                fmt, 64, 256, 8, 64, dtype, seed=310
            )

    class _FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, _ = _quantized_format_linear(
                fmt, 256, 256, 8, 64, dtype, seed=311
            )
            self.k_proj, _ = _quantized_format_linear(
                fmt, 64, 256, 8, 64, dtype, seed=312
            )
            self.v_proj, _ = _quantized_format_linear(
                fmt, 64, 256, 8, 64, dtype, seed=313
            )
            self.o_proj, _ = _quantized_format_linear(
                fmt, 256, 256, 8, 64, dtype, seed=314
            )
            self.indexer = _FakeIndexer()

    class _Mini(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _FakeAttn()

    with testing.verified("qwen4_exp"):
        model = _Mini()

    # 1. Fail closed when a row_exact_qmv warm future raises.
    monkeypatch.setattr(
        row_exact_qmv,
        "warm_model",
        lambda m, mtp_enabled=False: (_Future(RuntimeError("row_exact boom")),),
    )
    with pytest.raises(RuntimeError, match="row_exact boom"):
        qwen35_moe_routed_decode.warm_moe_model(model, mtp_enabled=False)

    monkeypatch.undo()

    # 2. Unrolled verification rejects a broken unrolled kernel (must not be
    # fooled by zero weights) and warms the rolled fallback instead.
    monkeypatch.setattr(row_exact_qmv, "_UNROLLED_VERDICTS", {})
    orig_build = row_exact_qmv.row_exact_skeleton.build_plan

    def broken_unrolled_build(faces, shapes, k, rows, dt, geometry, **kwargs):
        plan = orig_build(faces, shapes, k, rows, dt, geometry, **kwargs)
        if plan is not None and geometry[2]:
            # Return a wrapper whose launch returns zeros (bad unrolled bits)
            # while keeping plan.shape so unrolled_tile_verified executes.
            class _BadPlan:
                def __getattr__(self, name):
                    return getattr(plan, name)

                def __call__(self, x, operands=None):
                    return [mx.zeros(s, dtype=dt) for s in plan.output_shapes]

            return _BadPlan()
        return plan

    monkeypatch.setattr(
        row_exact_qmv.row_exact_skeleton, "build_plan", broken_unrolled_build
    )
    futs = row_exact_qmv.warm_model(model, mtp_enabled=True)
    for fut in futs:
        fut.result()
    assert row_exact_qmv._UNROLLED_VERDICTS, "expected unrolled verdicts populated"
    assert all(v is False for v in row_exact_qmv._UNROLLED_VERDICTS.values())


def test_warm_model_dispatch_uses_model_dtype_over_fixed_format_scale_dtype(
    monkeypatch,
):
    """pre-a1-aa9d16b9 MEDIUM 1: fixed-dtype format scales do not override model dtype."""
    from omlx.formats import QuantFormat

    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    monkeypatch.setattr(kernels, "_PROOFS", {})

    base = affine.AffineFormat()

    class _FixedFp16ScaleFormat(QuantFormat):
        mode = "fixed_fp16_scales"
        version = 1
        codes_name = "weight"
        tensors = ("scales", "biases")

        def layout(self, k, n, bits, group_size):
            return {
                "weight": ((n, k * bits // 32), mx.uint32),
                "scales": ((n, k // group_size), mx.float16),
                "biases": ((n, k // group_size), mx.float16),
            }

        def dot(self, layer, k, n):
            from dataclasses import replace

            return replace(base.dot(layer, k, n), dtypes=(mx.bfloat16, mx.float16))

    fmt = _FixedFp16ScaleFormat()
    k, n = 256, 64
    layout = fmt.layout(k, n, 8, 64)
    tensors = {
        "weight": mx.zeros(layout["weight"][0], mx.uint32),
        "scales": mx.zeros(layout["scales"][0], mx.float16),
        "biases": mx.zeros(layout["biases"][0], mx.float16),
    }

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            # Put format layer BEFORE the bfloat16 nn.Linear so _infer_model_dtype
            # sees the format layer first!
            self.fmt_lin = FormatLinear(
                fmt,
                8,
                64,
                k,
                tensors,
                None,
                placeholders=False,
                _load_layout=layout,
            )
            self.dense = nn.Linear(k, n, bias=False)
            self.dense.weight = mx.zeros((n, k), dtype=mx.bfloat16)

    with testing.verified("qwen4_exp"):
        model = _Model()

    assert model.fmt_lin._stored_floats == ()
    kept = model.fmt_lin._keep()
    assert kept.plan is not None
    kept.plan.proof.ran.add(mx.bfloat16)

    linear.warm_model_dispatch(model, prefill_lengths=(64,), mtp_enabled=False)
    assert ((1, 1, k), mx.bfloat16) in kept.plan.proof.graphs
    assert ((1, 1, k), mx.float16) not in kept.plan.proof.graphs


def test_register_fast_concurrent_creation_keeps_sigs_in_sync(monkeypatch):
    """pre-a1-aa9d16b9 MEDIUM 2 & Mutant 3: concurrent _register_fast + 64-entry clear."""
    import concurrent.futures
    import threading

    from omlx.custom_kernels.quant_matmul import fast as native

    if not mx.metal.is_available() or not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires Metal and FormatLinearFast native")

    fmt = affine.AffineFormat()
    dtype = mx.bfloat16
    with testing.verified("qwen4_exp"):
        layer, _ = _quantized_format_linear(fmt, 256, 256, 8, 64, dtype, seed=320)
    kept = layer._keep()
    kept.dense = layer._dense_tile(kept)
    assert kept.dense is not None

    x_a = mx.zeros((1, 64, 256), dtype=dtype)
    x_b = mx.zeros((1, 65, 256), dtype=dtype)
    sig_a = (x_a.shape, dtype)
    sig_b = (x_b.shape, dtype)
    kept.dense.binding.kept.graphs[sig_a] = lambda z, *ops: z
    kept.dense.binding.kept.graphs[sig_b] = lambda z, *ops: z

    # 1. Force two threads into _register_fast concurrently; if unlocked, both
    # call native.format_linear_fast and hit the barrier; if locked, they
    # serialize and the second reuses kept.dense_fast without calling it again.
    orig_ctor = native.format_linear_fast
    ctor_calls = 0
    ctor_barrier = threading.Barrier(2, timeout=0.5)

    def racing_ctor(*args, **kwargs):
        nonlocal ctor_calls
        ctor_calls += 1
        with __import__("contextlib").suppress(threading.BrokenBarrierError):
            ctor_barrier.wait()
        return orig_ctor(*args, **kwargs)

    monkeypatch.setattr(native, "format_linear_fast", racing_ctor)
    start_barrier = threading.Barrier(2)

    def worker(sig):
        kept.dense.binding.kept.graphs[sig] = lambda z, *ops: z
        start_barrier.wait()
        layer._register_fast(
            kept, kept.dense, sig, proof_state=kept.dense.binding.state
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(worker, sig_a)
        f2 = pool.submit(worker, sig_b)
        f1.result()
        f2.result()

    assert (
        ctor_calls == 1
    ), f"expected serialized single fast creation, got {ctor_calls}"
    assert layer._fast is not None
    # Main thread also has both graphs populated (line 2060 above), so replay
    # checks both registered signatures on the shared FormatLinearFast.
    assert layer._fast(layer, x_a) is not None
    assert layer._fast(layer, x_b) is not None
    assert kept.dense_fast_sigs == {sig_a, sig_b}

    # 2. Mutant 3 (linear.py:2468-2469): len(sigs) == 64 clears on the 65th sig
    # (mirroring format_dispatch.cpp:42), killing `== 64 -> == 65`.
    monkeypatch.setattr(native, "format_linear_fast", orig_ctor)
    kept.dense_fast = None
    kept.dense_fast_sigs = None
    for i in range(64):
        sig_i = ((1, 64 + i, 256), dtype)
        kept.dense.binding.kept.graphs[sig_i] = lambda z, *ops: z
        layer._register_fast(
            kept, kept.dense, sig_i, proof_state=kept.dense.binding.state
        )
    assert len(kept.dense_fast_sigs) == 64
    sig_64 = ((1, 128, 256), dtype)
    kept.dense.binding.kept.graphs[sig_64] = lambda z, *ops: z
    layer._register_fast(kept, kept.dense, sig_64, proof_state=kept.dense.binding.state)
    assert kept.dense_fast_sigs == {sig_64}


def test_warm_model_dispatch_rows_backend_does_not_bind_wide_fast_on_non_qwen(
    monkeypatch,
):
    """pre-a1-aa9d16b9 LOW 3: rows-backend wide warm never binds _fast on non-Qwen."""
    from omlx.formats import tile_dense, tile_wide

    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    monkeypatch.setattr(kernels, "_PROOFS", {})

    fmt = affine.AffineFormat()
    dtype = mx.bfloat16
    with testing.verified("qwen4_exp"):
        non_qwen, _ = _quantized_format_linear(fmt, 256, 256, 4, 64, dtype, seed=330)
        qwen_layer, _ = _quantized_format_linear(fmt, 256, 256, 4, 64, dtype, seed=331)
    qwen_layer._qwen_row_exact = lambda: False

    class _Wrap(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

    rows_policy = tile_dense.DenseTilePolicy(
        "rows",
        16,
        16,
        16,
        1,
        1,
        False,
        1,
        1,
        tile_dense.ParamStaging.PER_STEP,
        1,
        0,
        0,
        "seg",
    )
    steel_policy = tile_dense._default_steel_policy(2048, 256, 256)

    # 1. Non-Qwen layer with backend="rows" at length 48 -> must NOT bind _fast.
    kept_nq = non_qwen._keep()
    kept_nq.dense = non_qwen._dense_tile(kept_nq)
    assert kept_nq.dense is not None
    kept_nq.dense = tile_dense.DensePlan(
        face=kept_nq.dense.face,
        operands=kept_nq.dense.operands,
        proof=kept_nq.dense.proof,
        said=kept_nq.dense.said,
        schedule=((None, rows_policy),),
        schedules={dtype: ((None, rows_policy),)},
        k=256,
        n=256,
        bits=4,
        group_size=64,
        row_limit=kept_nq.dense.row_limit,
        binding=kept_nq.dense.binding,
    )
    wide_nq = linear._ordinary_tile(non_qwen, 256, 256, "wide", tile_wide.kernel)
    assert wide_nq is not None
    kept_nq.routes["wide"] = wide_nq
    kept_nq.wide = wide_nq
    metal_16, _ = tile_wide.describe(wide_nq, mx.zeros((16, 256), dtype))
    wide_nq.proof.ran.add(metal_16)
    monkeypatch.setattr(linear, "traversal", lambda *a, **kw: ("wide", None))
    monkeypatch.setattr(linear, "_metal_architecture", lambda: "applegpu_g15d")

    linear.warm_model_dispatch(
        _Wrap(non_qwen), prefill_lengths=(48,), mtp_enabled=False
    )
    assert non_qwen._fast is None, "non-Qwen rows-backend must not bind wide_fast"
    assert ((16, 256), dtype) in wide_nq.proof.graphs

    # Direct _project_route on non-Qwen must also not bind _fast.
    y_nq = non_qwen._project_route(
        kept_nq, mx.zeros((16, 256), dtype), ((16, 256), dtype), ("wide", None)
    )
    assert y_nq is not None
    assert non_qwen._fast is None

    # 2. Qwen layer with prefills=(2048, 48): length 48 must NOT leave _fast on
    # wide_fast when dense_fast was bound for 2048.
    kept_q = qwen_layer._keep()
    kept_q.dense = qwen_layer._dense_tile(kept_q)
    assert kept_q.dense is not None
    kept_q.dense = tile_dense.DensePlan(
        face=kept_q.dense.face,
        operands=kept_q.dense.operands,
        proof=kept_q.dense.proof,
        said=kept_q.dense.said,
        schedule=((64, rows_policy), (None, steel_policy)),
        schedules={dtype: ((64, rows_policy), (None, steel_policy))},
        k=256,
        n=256,
        bits=4,
        group_size=64,
        row_limit=kept_q.dense.row_limit,
        binding=kept_q.dense.binding,
    )
    eff_steel = tile_dense._effective_policy(steel_policy, 256, 256)
    kept_q.dense.proof.ran.add(
        tile_dense._metal_key(kept_q.dense, mx.zeros((1, 2048, 256), dtype), eff_steel)
    )
    wide_q = linear._ordinary_tile(qwen_layer, 256, 256, "wide", tile_wide.kernel)
    assert wide_q is not None
    kept_q.routes["wide"] = wide_q
    kept_q.wide = wide_q
    wide_q.proof.ran.add(metal_16)

    linear.warm_model_dispatch(
        _Wrap(qwen_layer), prefill_lengths=(2048, 48), mtp_enabled=False
    )
    assert kept_q.dense_fast is not None
    assert qwen_layer._fast is kept_q.dense_fast


def test_engine_prefill_lengths_includes_grid_aligned_steady_step():
    """pre-a1-aa9d16b9 LOW 4: steady-state 4096 step at chunk 2 is warmed."""

    class _GridSched:
        def _base_prefill_step_size(self, processed, _remaining):
            return 2048 if processed in (0, 2048) else 4096

    assert qwen35_moe_routed_decode.engine_prefill_lengths(_GridSched()) == (
        2047,
        2048,
        4095,
        4096,
    )


def test_aa9d16b9_surviving_mutants(monkeypatch):
    """Kill surviving mutants 1, 2, 4, 5, 6 from reviews/pre-a1-aa9d16b9.md."""
    from omlx.formats import tile_dense, tile_wide, tuning

    if not mx.metal.is_available():
        pytest.skip("requires Metal")

    fmt = affine.AffineFormat()
    dtype = mx.bfloat16

    # Mutant 1 (qwen35_moe_routed_decode.py:1976-1977): non-empty switch_mlp
    # with num_experts=0 or num_experts=True must NOT call warm_existing_switch.
    warmed_switches = []
    monkeypatch.setattr(
        linear,
        "warm_existing_switch",
        lambda layer, **kw: warmed_switches.append(layer) or (),
    )
    monkeypatch.setattr(kernels, "warm", lambda _inst: ())
    monkeypatch.setattr(kernels, "warm_proofs", lambda _tasks: ())

    class _BadExpertsBlock(nn.Module):
        def __init__(self, num_experts, seed):
            super().__init__()
            self.top_k = 2
            self.num_experts = num_experts
            self.switch_mlp = nn.Module()
            self.switch_mlp.gate_up_proj, _ = _quantized_format_switch(
                fmt, 4, 128, 64, 4, 64, dtype, seed=seed
            )

    class _BadExpertsContainer(nn.Module):
        def __init__(self):
            super().__init__()
            self.b0 = _BadExpertsBlock(0, seed=340)
            self.bb = _BadExpertsBlock(True, seed=341)

    with testing.verified("qwen4_exp"):
        bad = _BadExpertsContainer()
    qwen35_moe_routed_decode.warm_moe_model(bad, mtp_enabled=False, warm_dispatch=False)
    assert warmed_switches == [], "invalid num_experts blocks must be skipped"
    monkeypatch.undo()

    # Mutant 2 (linear.py:2531): 1-row call on Qwen FormatLinear must NOT probe
    # wide.proof.graphs (guards `x.size > plan.k`).
    with testing.verified("qwen4_exp"):
        q_lin, _ = _quantized_format_linear(fmt, 256, 256, 8, 64, dtype, seed=342)
    q_lin._qwen_row_exact = lambda: False
    kept_q = q_lin._keep()
    wide_probes = []

    class _SpyGraphs(dict):
        def get(self, key, default=None):
            wide_probes.append(key)
            return super().get(key, default)

    kept_q.wide = SimpleNamespace(
        proof=SimpleNamespace(graphs=_SpyGraphs()), operands=()
    )
    y1 = q_lin(mx.zeros((1, 1, 256), dtype=dtype))
    mx.eval(y1)
    assert wide_probes == [], "1-row decode must not probe kept.wide.proof.graphs"

    # Mutant 4 (tile_dense.py:1197): instance attribute `_ordinary_serves` in
    # layer.__dict__ must bypass `_schedule_cache`.
    with testing.verified("qwen4_exp"):
        d_a, _ = _quantized_format_linear(fmt, 256, 512, 8, 64, dtype, seed=343)
        d_inst, _ = _quantized_format_linear(fmt, 256, 512, 8, 64, dtype, seed=344)
    d_a._keep()
    d_inst._keep()
    d_inst.__dict__["_ordinary_serves"] = lambda rows, *, chip=None: False
    sched_cache: dict = {}
    tile_dense.build_plan(d_a, tensor_units=False, _schedule_cache=sched_cache)
    inst_lookups = []
    orig_lookup = tuning.lookup
    monkeypatch.setattr(
        tuning,
        "lookup",
        lambda *a, **kw: (inst_lookups.append(a) or orig_lookup(*a, **kw)),
    )
    tile_dense.build_plan(d_inst, tensor_units=False, _schedule_cache=sched_cache)
    assert (
        len(inst_lookups) > 0
    ), "instance _ordinary_serves must bypass _schedule_cache"
    monkeypatch.undo()

    # Mutant 5 (linear.py:1452): b in kept.mlx_rows must be skipped by
    # warm_model_dispatch.
    with testing.verified("qwen4_exp"):
        mlx_lin, _ = _quantized_format_linear(fmt, 256, 256, 8, 64, dtype, seed=345)
    mlx_lin._qwen_row_exact = lambda: False
    kept_m = mlx_lin._keep()
    kept_m.mlx_rows = frozenset({2})
    traversed_b = []
    monkeypatch.setattr(linear, "_metal_architecture", lambda: "applegpu_g15d")
    monkeypatch.setattr(
        linear,
        "traversal",
        lambda b, *a, **kw: (traversed_b.append(b) or ("rows", None)),
    )

    class _One(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

    linear.warm_model_dispatch(
        _One(mlx_lin), decode_batch_sizes=(2, 4), prefill_lengths=(), mtp_enabled=False
    )
    assert traversed_b == [4], f"expected b=2 skipped via mlx_rows, got {traversed_b}"
    monkeypatch.undo()

    # Mutant 6 (tile_dense.py:1696): unproven wide Metal must NOT be kept by
    # tile_dense.warm_graph.
    monkeypatch.setattr(kernels, "_PROOFS", {})
    with testing.verified("qwen4_exp"):
        w_lin, _ = _quantized_format_linear(fmt, 256, 256, 8, 64, dtype, seed=346)
    base_plan = tile_dense.build_plan(w_lin, tensor_units=False)
    assert base_plan is not None
    wide_policy = tile_dense.DenseTilePolicy(
        "wide",
        16,
        16,
        16,
        1,
        1,
        False,
        1,
        1,
        tile_dense.ParamStaging.PER_STEP,
        1,
        0,
        0,
        "seg",
    )
    unproven_wide_proof = kernels.proof("test_unproven_wide", base_plan.face)
    forced_wide = tile_dense.DensePlan(
        face=base_plan.face,
        operands=base_plan.operands,
        proof=unproven_wide_proof,
        said=base_plan.said,
        schedule=((None, wide_policy),),
        schedules={dtype: ((None, wide_policy),)},
        k=base_plan.k,
        n=base_plan.n,
        bits=base_plan.bits,
        group_size=base_plan.group_size,
        row_limit=base_plan.row_limit,
        binding=unproven_wide_proof.bind(),
    )
    assert tile_wide.kernel(base_plan.face) is not None
    assert tile_dense.warm_graph(forced_wide, (1, 16, 256), dtype) is False
    assert forced_wide.proof.graphs == {}
    assert forced_wide.binding.kept.graphs == {}


def test_warm_row_exact_qmv_prewarms_all_reference_format_linears(monkeypatch):
    """pre-a1-d090ff51 MEDIUM 1: row_exact_qmv.warm_model pre-warms all FormatLinear
    reference projections via Proof.warm (zero Proof.prove / proof_count bumps,
    fail-closed on failed proof, no mid-loop orphaned futures)."""
    from omlx.patches import row_exact_qmv

    if not mx.metal.is_available():
        pytest.skip("requires Metal")

    monkeypatch.setattr(kernels, "_PROOFS", {})
    monkeypatch.setattr(row_exact_qmv, "_UNROLLED_VERDICTS", {})
    fmt = affine.AffineFormat()
    dtype = mx.bfloat16

    class _FakeIndexer(nn.Module):
        def __init__(self):
            super().__init__()
            self.index_qk_proj, _ = _quantized_format_linear(
                fmt, 64, 256, 8, 64, dtype, seed=350
            )

    class _FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, _ = _quantized_format_linear(
                fmt, 128, 256, 8, 64, dtype, seed=351
            )
            self.k_proj, _ = _quantized_format_linear(
                fmt, 64, 256, 8, 64, dtype, seed=352
            )
            self.v_proj, _ = _quantized_format_linear(
                fmt, 64, 256, 8, 64, dtype, seed=353
            )
            self.o_proj, _ = _quantized_format_linear(
                fmt, 256, 256, 8, 64, dtype, seed=354
            )
            self.indexer = _FakeIndexer()

    class _Mini(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _FakeAttn()

    with testing.verified("qwen4_exp"):
        model = _Mini()

    # 1. Standalone warm_model(mtp_enabled=True) must pre-warm q/k/v/idx/o
    #    1-row FormatLinear Metals via Proof.warm (0 proof_count delta).
    before = kernels.proof_count()
    for fut in row_exact_qmv.warm_model(model, mtp_enabled=True):
        fut.result()
    assert (
        kernels.proof_count() == before
    ), "row_exact_qmv.warm_model triggered Proof.prove() instead of Proof.warm()"
    for proj in (
        model.attn.q_proj,
        model.attn.k_proj,
        model.attn.v_proj,
        model.attn.indexer.index_qk_proj,
        model.attn.o_proj,
    ):
        s_plan = linear._plan(proj)
        assert s_plan is not None and dtype in s_plan.proof.ran

    # 2. Failed reference proof must fail closed (not silently skipped).
    monkeypatch.setattr(row_exact_qmv, "_UNROLLED_VERDICTS", {})
    q_plan = linear._plan(model.attn.q_proj)
    assert q_plan is not None
    q_plan.proof.ran.clear()
    q_plan.proof.failed = True
    with pytest.raises(
        RuntimeError, match="cannot warm a proof that has already failed"
    ):
        row_exact_qmv.warm_model(model, mtp_enabled=True)

    # 3. Mid-loop scan error must not submit/orphan any warm_proofs futures.
    q_plan.proof.failed = False
    submitted: list = []
    orig_warm_proofs = kernels.warm_proofs
    monkeypatch.setattr(
        kernels,
        "warm_proofs",
        lambda tasks: (submitted.append(tasks) or orig_warm_proofs(tasks)),
    )

    class _BoomModel:
        def named_modules(self):
            yield ("attn", model.attn)
            raise RuntimeError("mid-loop scan boom")

    with pytest.raises(RuntimeError, match="mid-loop scan boom"):
        row_exact_qmv.warm_model(_BoomModel(), mtp_enabled=True)
    assert submitted == [], "warm_proofs must not be submitted before scan completes"


def test_warm_row_exact_qmv_handles_non_int_group_size(monkeypatch):
    """pre-a1-d090ff51 LOW 2: non-int group_size (None or tensor) must not crash
    warm_model, and has_idx=False + mtp_enabled=False skips attention work."""
    from omlx.formats import QuantFormat
    from omlx.patches import row_exact_qmv

    if not mx.metal.is_available():
        pytest.skip("requires Metal")

    class _NoGroupFormat(QuantFormat):
        mode = "no_group_test"
        version = 1
        codes_name = "weight"
        tensors = ("scales",)

        def layout(self, k, n, bits, group_size):
            return {
                "weight": ((n, k * bits // 32), mx.uint32),
                "scales": ((n, 1), mx.bfloat16),
            }

        def dot(self, layer, k, n):
            return None

    fmt = _NoGroupFormat()
    k, n = 256, 64
    layout = fmt.layout(k, n, 8, None)

    def _mk():
        tensors = {
            "weight": mx.zeros(layout["weight"][0], mx.uint32),
            "scales": mx.zeros(layout["scales"][0], mx.bfloat16),
        }
        return FormatLinear(
            fmt,
            8,
            None,
            k,
            tensors,
            None,
            placeholders=False,
            _load_layout=layout,
        )

    class _NoIdxAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = _mk()
            self.k_proj = _mk()
            self.v_proj = _mk()
            self.o_proj = _mk()

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _NoIdxAttn()

    with testing.verified("qwen4_exp"):
        model = _Model()
    assert row_exact_qmv.warm_model(model, mtp_enabled=False) == ()
    assert row_exact_qmv.warm_model(model, mtp_enabled=True) == ()

    # warm_model_dispatch must skip non-Qwen4 attention when has_idx=False & mtp=False.
    dtype_calls = []
    orig_attn_dt = row_exact_qmv._attn_warm_dtypes
    monkeypatch.setattr(
        row_exact_qmv,
        "_attn_warm_dtypes",
        lambda *a, **kw: (dtype_calls.append(a) or orig_attn_dt(*a, **kw)),
    )
    row_exact_qmv.warm_model_dispatch(model, mtp_enabled=False)
    assert dtype_calls == [], "has_idx=False & mtp_enabled=False must skip early"


def test_d090ff51_surviving_mutants(monkeypatch):
    """Kill all 6 surviving mutants + compile-time opt 1 from pre-a1-d090ff51.md."""
    import contextlib
    import math
    import threading
    import time
    from types import SimpleNamespace

    from omlx.formats import QuantFormat
    from omlx.patches import row_exact_qmv

    if not mx.metal.is_available():
        pytest.skip("requires Metal")

    fmt = affine.AffineFormat()
    dtype = mx.bfloat16

    # Mutant 1 (row_exact_qmv.py:1336): warm_model_dispatch populates _GROUP_PLANS
    # and GDN format_rows_qmv proof.graphs before request #0.
    class _FakeIndexer(nn.Module):
        def __init__(self):
            super().__init__()
            self.index_qk_proj, _ = _quantized_format_linear(
                fmt, 64, 256, 8, 64, dtype, seed=360
            )

    class _FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, _ = _quantized_format_linear(
                fmt, 128, 256, 8, 64, dtype, seed=361
            )
            self.k_proj, _ = _quantized_format_linear(
                fmt, 64, 256, 8, 64, dtype, seed=362
            )
            self.v_proj, _ = _quantized_format_linear(
                fmt, 64, 256, 8, 64, dtype, seed=363
            )
            self.o_proj, _ = _quantized_format_linear(
                fmt, 256, 256, 8, 64, dtype, seed=364
            )
            self.indexer = _FakeIndexer()

    class _AttnModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _FakeAttn()

    with testing.verified("qwen4_exp"):
        attn_model = _AttnModel()
    # Warm Metals first so warm_model_dispatch populates _group_plan cleanly.
    for fut in row_exact_qmv.warm_model(attn_model, mtp_enabled=False):
        fut.result()
    group_plans_before = row_exact_qmv._plans(attn_model.attn.q_proj)
    assert len(group_plans_before) == 0
    row_exact_qmv.warm_model_dispatch(attn_model, mtp_enabled=False)
    group_plans_after = row_exact_qmv._plans(attn_model.attn.q_proj)
    assert group_plans_after, "warm_model_dispatch must populate _group_plan cache"

    # Mutant 2 (qwen35_moe_routed_decode.py:1922): warm_engine_dispatch runs
    # row_exact_qmv.warm_model for layers enabled AFTER warm_moe_model (e.g.,
    # configure_qwen4_decode toggling _omlx_qwen4_wide_projections).
    warm_calls = []
    orig_rm_warm = row_exact_qmv.warm_model
    monkeypatch.setattr(
        row_exact_qmv,
        "warm_model",
        lambda m, mtp_enabled=False: (
            warm_calls.append((m, mtp_enabled))
            or orig_rm_warm(m, mtp_enabled=mtp_enabled)
        ),
    )
    qwen35_moe_routed_decode.warm_engine_dispatch(
        attn_model, mx.default_stream(mx.gpu), prefill_lengths=(), mtp_enabled=False
    )
    assert (
        attn_model,
        False,
    ) in warm_calls, "warm_engine_dispatch must invoke row_exact_qmv.warm_model"
    monkeypatch.undo()

    # Mutant 3 (row_exact_qmv.py:1039): when unrolled_tile_verified returns False,
    # the rolled fallback instance IS compiled into rolled.kernel._built.
    # Also verifies Opt 1: when unrolled_tile_verified freshly runs and returns True,
    # inst._build_locked is NOT called a second time, yet inst IS in kernel._built.
    monkeypatch.setattr(row_exact_qmv, "_UNROLLED_VERDICTS", {})
    rolled_built_plans = []
    orig_build = row_exact_qmv.row_exact_skeleton.build_plan

    def _capture_and_break(faces, shapes, k, rows, dt, geometry, **kwargs):
        plan = orig_build(faces, shapes, k, rows, dt, geometry, **kwargs)
        if plan is not None and not geometry[2]:
            rolled_built_plans.append((plan, k, dt))
        if plan is not None and geometry[2]:

            class _BadPlan:
                def __getattr__(self, name):
                    return getattr(plan, name)

                def __call__(self, x, operands=None):
                    return [mx.zeros(s, dtype=dt) for s in plan.output_shapes]

            return _BadPlan()
        return plan

    monkeypatch.setattr(
        row_exact_qmv.row_exact_skeleton, "build_plan", _capture_and_break
    )
    for fut in row_exact_qmv.warm_model(attn_model, mtp_enabled=True):
        fut.result()
    assert rolled_built_plans, "expected rolled fallback plans captured"
    fallback_checked = 0
    for plan, k_dim, dt in rolled_built_plans:
        in_specs, _ = row_exact_qmv._plan_stand_in(plan, k_dim, dt)
        inst = kernels.planned(
            plan.kernel, in_specs, plan.output_dtypes, template=plan.template
        )
        assert inst in inst.kernel._built, "rolled fallback instance must be built"
        fallback_checked += 1
    assert fallback_checked > 0
    monkeypatch.undo()

    # Opt 1: freshly verified unrolled tile marks inst in kernel._built without
    # invoking _Instance._build_locked a second time.
    monkeypatch.setattr(row_exact_qmv, "_UNROLLED_VERDICTS", {})
    unrolled_plans = []

    def _capture_unrolled(faces, shapes, k, rows, dt, geometry, **kwargs):
        plan = orig_build(faces, shapes, k, rows, dt, geometry, **kwargs)
        if plan is not None and geometry[2]:
            # Clear _built so a redundant build would call _build_locked.
            in_specs, _ = row_exact_qmv._plan_stand_in(plan, k, dt)
            inst = kernels.planned(
                plan.kernel, in_specs, plan.output_dtypes, template=plan.template
            )
            inst.kernel._built.clear()
            unrolled_plans.append((plan, k, dt))
        return plan

    monkeypatch.setattr(
        row_exact_qmv.row_exact_skeleton, "build_plan", _capture_unrolled
    )
    unrolled_build_calls = []
    orig_build_locked = kernels._Instance._build_locked

    def _spy_build_locked(self, describe):
        for p, k_dim, dt in unrolled_plans:
            in_specs, _ = row_exact_qmv._plan_stand_in(p, k_dim, dt)
            inst = kernels.planned(
                p.kernel, in_specs, p.output_dtypes, template=p.template
            )
            if self.kernel is inst.kernel:
                unrolled_build_calls.append(self)
        return orig_build_locked(self, describe)

    monkeypatch.setattr(kernels._Instance, "_build_locked", _spy_build_locked)
    for fut in row_exact_qmv.warm_model(attn_model, mtp_enabled=True):
        fut.result()
    assert unrolled_plans, "expected unrolled plans"
    assert (
        unrolled_build_calls == []
    ), f"freshly verified unrolled tiles must not re-run _build_locked: {unrolled_build_calls}"
    for plan, k_dim, dt in unrolled_plans:
        in_specs, _ = row_exact_qmv._plan_stand_in(plan, k_dim, dt)
        inst = kernels.planned(
            plan.kernel, in_specs, plan.output_dtypes, template=plan.template
        )
        assert inst in inst.kernel._built
    monkeypatch.undo()

    # Mutant 4 (row_exact_qmv.py:77-79): concurrent unrolled_tile_verified on the
    # same shape must serialize under the per-shape lock and run checks ONCE.
    monkeypatch.setattr(row_exact_qmv, "_UNROLLED_VERDICTS", {})
    monkeypatch.setattr(row_exact_qmv, "_UNROLLED_LOCKS", {})
    shape_key = (8, 64, 128, 256, 2)
    barrier = threading.Barrier(2, timeout=10)
    run_calls = 0
    count_lock = threading.Lock()
    results: list = []

    def _run_once(p):
        nonlocal run_calls
        time.sleep(0.02)
        with count_lock:
            run_calls += 1
        return [mx.zeros((2, 128), dtype=dtype)]

    def _ref_once(p):
        return [mx.zeros((2, 128), dtype=dtype)]

    def _worker():
        try:
            barrier.wait()
            results.append(
                row_exact_qmv.unrolled_tile_verified(
                    shape_key, _run_once, _ref_once, 2, 256, dtype
                )
            )
        except BaseException:
            with contextlib.suppress(threading.BrokenBarrierError):
                barrier.abort()
            raise

    t1 = threading.Thread(target=_worker)
    t2 = threading.Thread(target=_worker)
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)
    assert results == [True, True]
    assert run_calls == 1, f"expected per-shape lock to run check once, got {run_calls}"
    monkeypatch.undo()

    # Mutant 5 (row_exact_qmv.py:901): _plan_stand_in on a narrow 2D weight tensor
    # with < 4 elements/row (shape (32, 1), size=32 >= 8, 2*rps=2) must pad rows
    # to >= -(-8 * n_j // arr.size) = 8 so stand-in keeps size >= 8.
    narrow_plan = SimpleNamespace(
        template=(("ROWS", 2), ("RPS", 1)),
        output_shapes=((2, 32),),
        operand_keys=(("weight",),),
        operands=(mx.zeros((32, 1), dtype=mx.uint32),),
    )
    in_specs, _ = row_exact_qmv._plan_stand_in(narrow_plan, 256, dtype)
    stand_weight_shape, _ = in_specs[1]
    assert stand_weight_shape == (8, 1), f"expected (8, 1), got {stand_weight_shape}"
    assert math.prod(stand_weight_shape) >= 8

    # Mutant 6 (linear.py:1223-1228): pass-2 _tensors fallback in _infer_model_dtype
    # when a model has ONLY a FormatLinear with _stored_floats=() and no bias / nn.Linear.
    class _Pass2OnlyFormat(QuantFormat):
        mode = "pass2_only_test"
        version = 1
        codes_name = "weight"
        tensors = ("scales",)

        def layout(self, k, n, bits, group_size):
            return {
                "weight": ((n, k * bits // 32), mx.uint32),
                "scales": ((n, k // group_size), mx.float16),
            }

        def dot(self, layer, k, n):
            return None

    p2_fmt = _Pass2OnlyFormat()
    p2_layout = p2_fmt.layout(256, 64, 8, 64)
    with testing.verified("qwen4_exp"):
        p2_lin = FormatLinear(
            p2_fmt,
            8,
            64,
            256,
            {
                "weight": mx.zeros(p2_layout["weight"][0], mx.uint32),
                "scales": mx.zeros(p2_layout["scales"][0], mx.float16),
            },
            None,
            placeholders=False,
            _load_layout=p2_layout,
        )
    assert p2_lin._stored_floats == ()

    class _OnlyFmt(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = p2_lin

    assert linear._infer_model_dtype(_OnlyFmt()) == mx.float16
