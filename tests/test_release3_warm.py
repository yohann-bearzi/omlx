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
    steel = real_plan.schedule[-1].policy
    wide = tile_dense.DenseTilePolicy(backend="wide")
    custom_sched = (
        tile_dense.DenseRoute(32, steel.backend, steel),
        tile_dense.DenseRoute(None, "wide", wide),
    )
    custom_plan = tile_dense.DensePlan(
        face=real_plan.face,
        operands=real_plan.operands,
        proof=real_plan.proof,
        said=real_plan.said,
        k=real_plan.k,
        n=real_plan.n,
        bits=real_plan.bits,
        group_size=real_plan.group_size,
        schedule=custom_sched,
        schedules={mx.bfloat16: custom_sched},
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
    assert ("wide", mx.bfloat16, 256, 256, 5, False, 0) in wide_metals
    assert ("wide", mx.bfloat16, 256, 256, 5, True, 0) in wide_metals


def test_tile_dense_wide_warm_specs_covers_exact_5_and_7(monkeypatch):
    """LOW-1: load-time warming proves the wide EXACT_ROWS 5 and 7 modes."""
    from omlx.formats import tile_dense

    fmt = affine.AffineFormat()
    with testing.verified("qwen4_exp"):
        layer, _ = _quantized_format_linear(fmt, 256, 256, 4, 64, mx.bfloat16, seed=81)

    real_plan = tile_dense.build_plan(layer, tensor_units=False)
    assert real_plan is not None
    wide = tile_dense.DenseTilePolicy(backend="wide")
    sched = (tile_dense.DenseRoute(16, "wide", wide),)
    plan = tile_dense.DensePlan(
        face=real_plan.face,
        operands=real_plan.operands,
        proof=real_plan.proof,
        said=real_plan.said,
        k=real_plan.k,
        n=real_plan.n,
        bits=real_plan.bits,
        group_size=real_plan.group_size,
        schedule=sched,
        schedules={mx.bfloat16: sched},
        row_limit=real_plan.row_limit,
        binding=real_plan.binding,
    )
    monkeypatch.setattr(tile_dense, "build_plan", lambda _l, **_kw: plan)

    instances, proofs = tile_dense.warm_specs(layer, (mx.bfloat16,), tensor_units=False)
    by_metal = {}
    for inst, entries in proofs.items():
        for _proof, metal in entries:
            if isinstance(metal, tuple) and metal[0] == "wide":
                by_metal[metal] = inst
    five = ("wide", mx.bfloat16, 256, 256, 5, False, 5)
    seven = ("wide", mx.bfloat16, 256, 256, 4, True, 7)
    assert five in by_metal, sorted(by_metal)
    assert seven in by_metal, sorted(by_metal)
    assert instances[by_metal[five]]().grid == (32, 2, 1)
    assert instances[by_metal[seven]]().grid == (64, 2, 1)


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
    orig_register_dot_fast = linear.FormatLinear._register_dot_fast

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

    def spy_register_dot_fast(self, *args, **kwargs):
        register_fast_calls.append((self, args[1] if len(args) > 1 else None))
        return orig_register_dot_fast(self, *args, **kwargs)

    monkeypatch.setattr(linear._FormatLayer, "_keep", spy_keep)
    monkeypatch.setattr(tile_dense, "build_plan", spy_build_plan)
    monkeypatch.setattr(mx, "compile", spy_compile)
    monkeypatch.setattr(linear.FormatLinear, "_register_fast", spy_register_fast)
    monkeypatch.setattr(
        linear.FormatLinear, "_register_dot_fast", spy_register_dot_fast
    )

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
    # + disarmed Qwen verify (proj_b) & non-Qwen verify (proj_a) + switch + routed MoE.
    y_pre_a = model.proj_a(x_pre)
    y_pre_b = model.proj_b(x_pre)
    y_pre2_a = model.proj_a(x_pre2)
    y_pre_n1 = model.proj_n1(x_pre)
    y_dec_a = model.proj_a(x_dec)
    y_dec_b = model.proj_b(x_dec)
    y_b2_b = model.proj_b(x_b2)
    y_b4_b = model.proj_b(x_b4)
    y_ver_b = model.proj_b(x_ver)
    y_pre_b_again = model.proj_b(x_pre)
    y_dec_n1 = model.proj_n1(x_dec)
    y_ver_a = model.proj_a(x_ver)

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
        y_ver_b,
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
    model.proj_a.__dict__.pop("_fast2", None)
    model.proj_b.__dict__.pop("_omlx_format_call", None)
    model.proj_b.__dict__.pop("_fast", None)
    model.proj_b.__dict__.pop("_fast2", None)
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
        orig_register_dot_fast = linear.FormatLinear._register_dot_fast

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
        monkeypatch.setattr(
            linear.FormatLinear,
            "_register_dot_fast",
            _wrap("register_fast", orig_register_dot_fast),
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
        schedule=(tile_dense.DenseRoute(None, "steel", steel_policy),),
        schedules={dtype: (tile_dense.DenseRoute(None, "steel", steel_policy),)},
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
    """build_plan(_schedule_cache=...) matches uncached and keys on dot_ok."""
    from omlx.formats import tile_dense, tuning

    fmt = affine.AffineFormat()
    dtype = mx.bfloat16
    with testing.verified("qwen4_exp"):
        a, _ = _quantized_format_linear(fmt, 256, 512, 8, 64, dtype, seed=301)
        b, _ = _quantized_format_linear(fmt, 256, 512, 8, 64, dtype, seed=302)

    a._keep()
    b._keep()

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

    # A planless (dot_face=None) layer has dot_ok=False and must NOT hit a's entry.
    p_c = tile_dense.build_plan(
        b, tensor_units=False, dot_face=None, _schedule_cache=cache
    )
    assert len(lookups) > 0, "dot_face=None (dot_ok=False) must miss dot_ok=True cache"
    assert (
        p_uncached is not None
        and p_a is not None
        and p_b is not None
        and p_c is not None
    )
    assert p_a.schedules == p_uncached.schedules
    assert p_b.schedules == p_uncached.schedules
    assert (
        p_c.schedules
        == tile_dense.build_plan(b, tensor_units=False, dot_face=None).schedules
    )
    assert len(cache) == 2 * len(p_a.schedules)


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
    layer = FormatLinear(
        fmt,
        8,
        64,
        k,
        {"weight": w, "scales": s, "biases": b},
        None,
        placeholders=False,
        _load_layout=layout,
    )
    return layer


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

# 1-row GDN in/out + 1-row Attention group4/group1 + GDN batch decode (rows 2,4).
dec_plan = qwen35_gdn_prework._qwen4_decode_plan(model.gdn)
assert dec_plan is not None
y_gdn_in = dec_plan[-3](x1)
y_gdn_out = dec_plan[-2](xo1)
stacked_gdn = dec_plan[-1]
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
# Batch decode (2, 4 rows) uses stacked FormatLinear and out_proj FormatLinear.
from omlx.formats import kernels as _fmt_kernels

proofs_before = _fmt_kernels.proof_count()
xb2 = (mx.random.normal((1, 2, 2560)) * 0.2).astype(dtype)
xob2 = (mx.random.normal((1, 2, 6144)) * 0.2).astype(dtype)
mx.eval(xb2, xob2)
yb_in = stacked_gdn(xb2)
yb_out = model.gdn.out_proj(xob2)
mx.eval(y_gdn_in, y_gdn_out, *y_g4, *y_g1, yb_in, yb_out)

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
    # 1-row verify uses in_rows/out_rows; >=2 rows uses stacked/out FormatLinear.
    yv1_in = ver_plan[0](x1)
    yv1_out = ver_plan[1](xo1)
    mx.eval(yv1_in, yv1_out)
    for rows in (2, 3, 4):
        xr = (mx.random.normal((1, rows, 2560)) * 0.2).astype(dtype)
        xor = (mx.random.normal((1, rows, 6144)) * 0.2).astype(dtype)
        mx.eval(xr, xor)
        yv_in = ver_plan[2](xr)
        yv_out = ver_plan[3](xor)
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
        mx.eval(yv_in, yv_out, *yg4, *yg1)
        # Bitwise check row 0 against serial 1-row FormatLinear.
        ref_r0 = model.attn.o_proj(xor[:, :1, :])
        mx.eval(ref_r0)
        bitwise_ok &= bool(
            mx.array_equal(
                yg1[0][:, :1, :].view(mx.uint16), ref_r0.view(mx.uint16)
            ).item()
        )

proofs_delta = _fmt_kernels.proof_count() - proofs_before
print(
    json.dumps(
        {
            "recorded": recorded,
            "bitwise_ok": bitwise_ok,
            "proofs_delta": proofs_delta,
        }
    )
)
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
        assert (
            data["proofs_delta"] == 0
        ), f"mtp={mtp_flag}: unexpected cold Proof.prove() count={data['proofs_delta']}"
        req0 = data["recorded"]["request0"]
        load = data["recorded"]["load"]
        assert req0 == [], (
            f"mtp={mtp_flag}: compiled omlx_row_exact_qmv libraries during "
            f"request #0: {req0}"
        )
        assert len(load) >= 4, f"mtp={mtp_flag}: expected >=4 load compiles, got {load}"
        # Rule 29: Qwen4 disarmed row-exact verify -> never warm dead group3 or
        # dead GDN multi-row unrolled row_exact_qmv libraries.
        assert not any(
            "group3" in name for name in load
        ), f"mtp={mtp_flag}: disarmed qwen4 must not warm dead group3: {load}"

        # MLX 0.32.4 mangled template args with "__" (_u__), 0.32.5 uses "_" (_u_).
        def _is_unrolled(name: str) -> bool:
            return "_u__" in name or "_u_bfloat16" in name or "_u_half" in name

        if mtp_flag == 0:
            assert not any(
                _is_unrolled(name) for name in load
            ), f"mtp=0 should not compile verify libraries: {load}"
        else:
            assert any(
                _is_unrolled(name) for name in load
            ), f"missing _u verify in {load}"
            assert all(
                "group4" in name or "group1" in name
                for name in load
                if _is_unrolled(name)
            ), f"unexpected non-attention unrolled verify warm in {load}"


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
    """pre-a1-aa9d16b9 MEDIUM 2 & Mutant 3 (adapted to c3 lock-free _slots):
    concurrent _register_fast shares one winning FormatLinearFast via
    kept._slots.setdefault with zero lost signatures, and FormatLinearFast
    clears its 64-entry table on the 65th signature."""
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
    dense = layer._dense_tile(kept)
    assert dense is not None

    x_a = mx.zeros((1, 64, 256), dtype=dtype)
    x_b = mx.zeros((1, 65, 256), dtype=dtype)
    sig_a = (x_a.shape, dtype)
    sig_b = (x_b.shape, dtype)
    dense.binding.kept.graphs[sig_a] = lambda z, *ops: z
    dense.binding.kept.graphs[sig_b] = lambda z, *ops: z

    # 1. Force two threads into _register_fast concurrently; atomic
    # kept._slots.setdefault picks ONE winning FormatLinearFast and BOTH
    # threads register their signatures on it.
    orig_ctor = native.format_linear_fast
    ctor_barrier = threading.Barrier(2, timeout=0.5)

    def racing_ctor(*args, **kwargs):
        with __import__("contextlib").suppress(threading.BrokenBarrierError):
            ctor_barrier.wait()
        return orig_ctor(*args, **kwargs)

    monkeypatch.setattr(native, "format_linear_fast", racing_ctor)
    start_barrier = threading.Barrier(2)

    def worker(sig):
        dense.binding.kept.graphs[sig] = lambda z, *ops: z
        start_barrier.wait()
        layer._register_fast(kept, dense, sig)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(worker, sig_a)
        f2 = pool.submit(worker, sig_b)
        f1.result()
        f2.result()

    assert layer._fast is kept.dense_fast and layer._fast is not None
    assert layer._fast(layer, x_a) is not None
    assert layer._fast(layer, x_b) is not None

    # 2. C++ FormatLinearFast clears its 64-entry table on the 65th signature
    # (format_dispatch.cpp:41-43).
    monkeypatch.setattr(native, "format_linear_fast", orig_ctor)
    kept._slots.pop("dense_fast", None)
    layer.__dict__.pop(linear._FAST, None)
    layer.__dict__.pop(linear._FAST2, None)
    x_first = mx.zeros((1, 64, 256), dtype=dtype)
    for i in range(64):
        sig_i = ((1, 64 + i, 256), dtype)
        dense.binding.kept.graphs[sig_i] = lambda z, *ops: z
        layer._register_fast(kept, dense, sig_i)
    assert layer._fast(layer, x_first) is not None
    x_64 = mx.zeros((1, 128, 256), dtype=dtype)
    sig_64 = (x_64.shape, dtype)
    dense.binding.kept.graphs[sig_64] = lambda z, *ops: z
    layer._register_fast(kept, dense, sig_64)
    assert layer._fast(layer, x_64) is not None
    assert layer._fast(layer, x_first) is None


def test_warm_model_dispatch_rows_backend_does_not_bind_wide_fast_on_non_qwen(
    monkeypatch,
):
    """pre-a1-aa9d16b9 LOW 3 (adapted to c3 schedule + _fast/_fast2): warming
    both a dot route (rows=4 -> dot_fast) and a dense route (prefill ->
    dense_fast) publishes both into _fast/_fast2 without overwriting _fast."""
    from omlx.custom_kernels.quant_matmul import fast as native
    from omlx.formats import tile_dense, tuning

    if not mx.metal.is_available() or not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires Metal and FormatLinearFast native")
    monkeypatch.setattr(kernels, "_PROOFS", {})

    fmt = affine.AffineFormat()
    dtype = mx.bfloat16
    dot_pol = tuning.DenseTilePolicy("dot")
    monkeypatch.setattr(
        tuning,
        "lookup",
        lambda route, *a, rows=1, **kw: (
            dot_pol if route == "dense_tile" and rows <= 4 else None
        ),
    )
    monkeypatch.setattr(tuning, "row_bounds", lambda route, **kw: (4, 16))
    with testing.verified("qwen4_exp"):
        layer, _ = _quantized_format_linear(fmt, 256, 256, 4, 64, dtype, seed=330)

    class _Wrap(nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.layer = mod

    wrap = _Wrap(layer)
    for fut in linear.warm_existing_dense(wrap, dtypes=(dtype,)) or ():
        fut.result()
    linear.warm_model_dispatch(
        wrap, decode_batch_sizes=(1, 4), prefill_lengths=(64,), mtp_enabled=False
    )
    kept = layer.__dict__[linear._KEPT]
    assert kept.dot_fast is not None and kept.dense_fast is not None
    assert {layer._fast, layer._fast2} == {kept.dot_fast, kept.dense_fast}
    first_fast = layer._fast

    # Alternating dot (rows=4) and dense (rows=64) hits _fast/_fast2 directly
    # without overwriting _fast or calling _scheduled/_dot_rows.
    monkeypatch.setattr(
        FormatLinear,
        "_scheduled",
        lambda *a, **kw: pytest.fail("warm call missed _fast/_fast2"),
    )
    monkeypatch.setattr(
        FormatLinear,
        "_dot_rows",
        lambda *a, **kw: pytest.fail("warm dot call missed _fast/_fast2"),
    )
    y4 = layer(mx.zeros((4, 1, 256), dtype=dtype))
    y64 = layer(mx.zeros((1, 64, 256), dtype=dtype))
    mx.eval(y4, y64)
    assert layer._fast is first_fast
    assert tile_dense.route_for(kept.dense, 4, dtype).backend == "dot"


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

    # Mutant 2: 1-row call on FormatLinear stays on the Dot plan and never
    # invokes _scheduled / tile_dense.route_for.
    with testing.verified("qwen4_exp"):
        q_lin, _ = _quantized_format_linear(fmt, 256, 256, 8, 64, dtype, seed=342)
    q_lin._keep()
    monkeypatch.setattr(
        FormatLinear,
        "_scheduled",
        lambda *a, **kw: pytest.fail("1-row decode must not enter _scheduled"),
    )
    y1 = q_lin(mx.zeros((1, 1, 256), dtype=dtype))
    mx.eval(y1)
    monkeypatch.undo()

    # Mutant 4: distinct dot_ok (dot_face=None vs real dot_face) must not
    # collide in build_plan's _schedule_cache.
    with testing.verified("qwen4_exp"):
        d_a, _ = _quantized_format_linear(fmt, 256, 512, 8, 64, dtype, seed=343)
    d_a._keep()
    sched_cache: dict = {}
    tile_dense.build_plan(d_a, tensor_units=False, _schedule_cache=sched_cache)
    inst_lookups = []
    orig_lookup = tuning.lookup
    monkeypatch.setattr(
        tuning,
        "lookup",
        lambda *a, **kw: (inst_lookups.append(a) or orig_lookup(*a, **kw)),
    )
    tile_dense.build_plan(
        d_a, tensor_units=False, dot_face=None, _schedule_cache=sched_cache
    )
    assert len(inst_lookups) > 0, "dot_face=None must not collide in _schedule_cache"
    monkeypatch.undo()

    # Mutant 5: 1 in kept.mlx_rows skips 1-row rows-graph warming in
    # warm_model_dispatch.
    with testing.verified("qwen4_exp"):
        mlx_lin, _ = _quantized_format_linear(fmt, 256, 256, 8, 64, dtype, seed=345)
    kept_m = mlx_lin._keep()
    kept_m.mlx_rows = frozenset({1})
    warmed_rows_shapes = []
    orig_rows_warm = linear._warm_linear_rows_graph

    def spy_rows_warm(plan, x_shape, dt, *, key):
        warmed_rows_shapes.append(x_shape)
        return orig_rows_warm(plan, x_shape, dt, key=key)

    monkeypatch.setattr(linear, "_warm_linear_rows_graph", spy_rows_warm)

    class _One(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

    linear.warm_model_dispatch(
        _One(mlx_lin), decode_batch_sizes=(1,), prefill_lengths=(), mtp_enabled=False
    )
    assert (
        1,
        1,
        256,
    ) not in warmed_rows_shapes, (
        f"expected 1-row skipped via mlx_rows, got {warmed_rows_shapes}"
    )
    monkeypatch.undo()

    # Mutant 6: unproven wide, mma, and msg Metals must NOT be kept by
    # tile_dense.warm_graph.
    monkeypatch.setattr(kernels, "_PROOFS", {})
    with testing.verified("qwen4_exp"):
        w_lin, _ = _quantized_format_linear(fmt, 256, 512, 8, 64, dtype, seed=346)
    base_plan = tile_dense.build_plan(w_lin, tensor_units=False)
    assert base_plan is not None
    assert tile_wide.kernel(base_plan.face) is not None
    for backend, pol, rows in (
        ("wide", tile_dense.DenseTilePolicy("wide"), 16),
        ("mma", tuning.MmaTilePolicy(16, 32, 8), 8),
        ("msg", tuning.MsgTilePolicy(4), 4),
    ):
        unproven_proof = kernels.proof(f"test_unproven_{backend}", base_plan.face)
        route_sched = (tile_dense.DenseRoute(None, backend, pol),)
        forced = tile_dense.DensePlan(
            face=base_plan.face,
            operands=base_plan.operands,
            proof=unproven_proof,
            said=base_plan.said,
            schedule=route_sched,
            schedules={dtype: route_sched},
            k=base_plan.k,
            n=base_plan.n,
            bits=base_plan.bits,
            group_size=base_plan.group_size,
            row_limit=base_plan.row_limit,
            binding=unproven_proof.bind(),
        )
        assert tile_dense.warm_graph(forced, (1, rows, 512), dtype) is False
        assert forced.proof.graphs == {}
        assert forced.binding.kept.graphs == {}


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

    # 2. Failed reference proof must fail closed (not silently skipped),
    #    including when metal is STILL present in Proof.ran (pre-a1-529daae8 LOW 1).
    monkeypatch.setattr(row_exact_qmv, "_UNROLLED_VERDICTS", {})
    q_plan = linear._plan(model.attn.q_proj)
    assert q_plan is not None
    q_plan.proof.ran.clear()
    q_plan.proof.failed = True
    with pytest.raises(
        RuntimeError, match="cannot warm a proof that has already failed"
    ):
        row_exact_qmv.warm_model(model, mtp_enabled=True)

    # LOW 1: metal still in proof.ran when proof.fail() ran for another metal.
    monkeypatch.setattr(row_exact_qmv, "_UNROLLED_VERDICTS", {})
    q_plan.proof.failed = False
    q_plan.proof.ran.add(dtype)
    q_plan.proof.fail()
    assert dtype in q_plan.proof.ran and q_plan.proof.failed
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

    # warm_model and warm_model_dispatch must skip non-indexer attention when
    # row-exact verify is not enabled (both mtp=False and mtp=True on release/4).
    dtype_calls = []
    orig_attn_dt = row_exact_qmv._attn_warm_dtypes
    monkeypatch.setattr(
        row_exact_qmv,
        "_attn_warm_dtypes",
        lambda *a, **kw: (dtype_calls.append(a) or orig_attn_dt(*a, **kw)),
    )
    row_exact_qmv.warm_model(model, mtp_enabled=False)
    row_exact_qmv.warm_model(model, mtp_enabled=True)
    row_exact_qmv.warm_model_dispatch(model, mtp_enabled=False)
    row_exact_qmv.warm_model_dispatch(model, mtp_enabled=True)
    assert dtype_calls == [], "has_idx=False without row-exact verify must skip early"


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


def test_r4_vfast_c2_gdn_stacked_and_disarmed_verify_warm_and_mutants(monkeypatch):
    """Release/4 VFAST c2 adaptations + pre-a1-529daae8 surviving mutants:
    1) GDN stacked FormatLinear (rows >= 2 batch decode & disarmed verify) warms
       Metals, plans, and dispatch at load with 0 cold proofs/plans/compiles;
    2) Disarmed Qwen4 verify (1, S, K) warms batched Qwen routes (not serial
       _linear), while non-Qwen layers keep serial (1, S, K);
    3) Rule 29: g3 attention warming runs ONLY when _omlx_mtp_row_exact_verify;
    4) Kills pre-a1-529daae8 surviving mutants 1-3 (GDN dispatch, unaligned-N
       ref_proof_tasks, pre-populated _UNROLLED_VERDICTS inst.build).
    """
    from omlx.formats import tile_dense
    from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
    from omlx.patches import qwen35_gdn_prework, row_exact_qmv

    if not mx.metal.is_available():
        pytest.skip("requires Metal")

    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import (
        Qwen4ExpGatedDeltaNet,
        Qwen4ExpRMSNormGated,
    )

    fmt = affine.AffineFormat()
    dtype = mx.bfloat16

    def _make_qwen_fl(n, k, seed):
        layer, _ = _quantized_format_linear(fmt, n, k, 8, 64, dtype, seed=seed)
        return layer

    # --- 1. GDN stacked FormatLinear + 1-row GDN dispatch (kills mutant 1) ---
    hk, hv, dk, dv = 16, 48, 128, 128
    c_dim = 2 * hk * dk + hv * dv

    def _make_gdn():
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
        mod.in_proj_qkv = _make_qwen_fl(c_dim, 2560, 411)
        mod.in_proj_z = _make_qwen_fl(hv * dv, 2560, 412)
        mod.in_proj_b = _make_qwen_fl(hv, 2560, 413)
        mod.in_proj_a = _make_qwen_fl(hv, 2560, 414)
        mod.out_proj = _make_qwen_fl(2560, hv * dv, 415)
        mod.eval()
        mx.eval(mod.parameters())
        return mod

    class _GdnModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gdn = _make_gdn()

    monkeypatch.setattr(kernels, "_PROOFS", {})
    with testing.verified("qwen4_exp"):
        gdn_model = _GdnModel()
        for fut in linear.warm_existing_dense(gdn_model, dtypes=(dtype,)) or ():
            fut.result()

    qwen35_moe_routed_decode.warm_moe_model(
        gdn_model, mtp_enabled=True, warm_dispatch=False
    )
    qwen35_moe_routed_decode.warm_engine_dispatch(
        gdn_model,
        mx.default_stream(mx.gpu),
        prefill_lengths=(),
        mtp_enabled=True,
    )

    dec_plan = qwen35_gdn_prework._qwen4_decode_plan(gdn_model.gdn)
    ver_plan = qwen35_gdn_prework._qwen4_verify_plan(gdn_model.gdn)
    assert dec_plan is not None and ver_plan is not None
    in_qmv, out_qmv, stacked = dec_plan[-3], dec_plan[-2], dec_plan[-1]
    assert isinstance(stacked, FormatLinear)
    assert linear._KEPT in stacked.__dict__, "stacked FormatLinear plan not prepared"

    # Spy all cold work during first request on GDN (1-row + rows 2,3,4).
    cold_events: list[str] = []
    orig_keep = linear._FormatLayer._keep
    orig_bp = tile_dense.build_plan
    orig_compile = mx.compile
    orig_rf = linear.FormatLinear._register_fast
    orig_rdf = linear.FormatLinear._register_dot_fast
    monkeypatch.setattr(
        linear._FormatLayer,
        "_keep",
        lambda self: (cold_events.append("keep") or orig_keep(self)),
    )
    monkeypatch.setattr(
        tile_dense,
        "build_plan",
        lambda *a, **kw: (cold_events.append("build_plan") or orig_bp(*a, **kw)),
    )
    monkeypatch.setattr(
        mx,
        "compile",
        lambda *a, **kw: (cold_events.append("compile") or orig_compile(*a, **kw)),
    )
    monkeypatch.setattr(
        linear.FormatLinear,
        "_register_fast",
        lambda self, *a, **kw: (
            cold_events.append("register_fast") or orig_rf(self, *a, **kw)
        ),
    )
    monkeypatch.setattr(
        linear.FormatLinear,
        "_register_dot_fast",
        lambda self, *a, **kw: (
            cold_events.append("register_dot_fast") or orig_rdf(self, *a, **kw)
        ),
    )

    proofs_before = kernels.proof_count()
    x1 = mx.zeros((1, 1, 2560), dtype=dtype)
    xo1 = mx.zeros((1, 1, 6144), dtype=dtype)
    outs = [in_qmv(x1), out_qmv(xo1), ver_plan[0](x1), ver_plan[1](xo1)]
    for rows in (2, 3, 4):
        outs.append(stacked(mx.zeros((1, rows, 2560), dtype=dtype)))
        outs.append(gdn_model.gdn.out_proj(mx.zeros((1, rows, 6144), dtype=dtype)))
    mx.eval(*outs)
    assert kernels.proof_count() == proofs_before
    assert cold_events == [], f"unexpected cold work on GDN request #0: {cold_events}"
    monkeypatch.undo()

    # --- 2. Construction-time schedule warms dense_fast (mlx) and dot_fast (dot) ---
    # Test in separate models/proofs since identical shapes share one Proof.
    from omlx.formats import tuning

    class _SingleModel(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.proj = layer

    monkeypatch.setattr(kernels, "_PROOFS", {})
    with testing.verified("qwen4_exp"):
        q_lin = _make_qwen_fl(256, 512, 421)
    q_model = _SingleModel(q_lin)
    for fut in linear.warm_existing_dense(q_model, dtypes=(dtype,)) or ():
        fut.result()
    linear.warm_model_dispatch(
        q_model, prefill_lengths=(), decode_batch_sizes=(1, 2, 4), mtp_enabled=True
    )
    q_kept = q_lin.__dict__[linear._KEPT]
    assert ((1, 3, 512), dtype) not in q_kept.plan.proof.graphs
    assert ((1, 3, 512), dtype) in q_kept.dense.binding.kept.graphs
    # And scheduled verify (1, 3, 512) must run with 0 cold proofs/compiles.
    q_before = kernels.proof_count()
    y_qv = q_lin(mx.zeros((1, 3, 512), dtype=dtype))
    mx.eval(y_qv)
    assert kernels.proof_count() == q_before

    monkeypatch.setattr(kernels, "_PROOFS", {})
    dot_pol = tuning.DenseTilePolicy("dot")
    with monkeypatch.context() as dot_patch:
        dot_patch.setattr(
            tuning,
            "lookup",
            lambda route, *a, rows=1, **kw: (
                dot_pol if route == "dense_tile" and rows <= 4 else None
            ),
        )
        dot_patch.setattr(tuning, "row_bounds", lambda route, **kw: (4, 16))
        with testing.verified("qwen4_exp"):
            nq_lin, _ = _quantized_format_linear(fmt, 256, 512, 8, 64, dtype, seed=422)
        nq_model = _SingleModel(nq_lin)
        for fut in linear.warm_existing_dense(nq_model, dtypes=(dtype,)) or ():
            fut.result()
        linear.warm_model_dispatch(
            nq_model, prefill_lengths=(), decode_batch_sizes=(1, 2, 4), mtp_enabled=True
        )
    nq_kept = nq_lin.__dict__[linear._KEPT]
    assert ((1, 3, 512), dtype) in nq_kept.plan.proof.graphs
    nq_before = kernels.proof_count()
    y_nqv = nq_lin(mx.zeros((1, 3, 512), dtype=dtype))
    mx.eval(y_nqv)
    assert kernels.proof_count() == nq_before

    # --- 3. Rule 29: g3 warmed ONLY when _omlx_mtp_row_exact_verify is True ---
    class _FakeIndexer(nn.Module):
        def __init__(self):
            super().__init__()
            self.index_qk_proj = _make_qwen_fl(64, 256, 430)

    class _FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = _make_qwen_fl(128, 256, 431)
            self.k_proj = _make_qwen_fl(64, 256, 432)
            self.v_proj = _make_qwen_fl(64, 256, 433)
            self.o_proj = _make_qwen_fl(256, 256, 434)
            self.indexer = _FakeIndexer()

    class _DisarmedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _FakeAttn()

    class _RowExactModel(nn.Module):
        _omlx_mtp_row_exact_verify = True

        def __init__(self):
            super().__init__()
            self.attn = _FakeAttn()

    with testing.verified("qwen4_exp"):
        disarmed_m = _DisarmedModel()
        row_exact_m = _RowExactModel()

    for fut in row_exact_qmv.warm_model(disarmed_m, mtp_enabled=True):
        fut.result()
    row_exact_qmv.warm_model_dispatch(disarmed_m, mtp_enabled=True)
    disarmed_keys = list(row_exact_qmv._plans(disarmed_m.attn.q_proj).keys())
    assert disarmed_keys and all(
        k[0] != 3 for k in disarmed_keys
    ), f"disarmed Qwen4 must not plan group3: {disarmed_keys}"

    for fut in row_exact_qmv.warm_model(row_exact_m, mtp_enabled=True):
        fut.result()
    row_exact_qmv.warm_model_dispatch(row_exact_m, mtp_enabled=True)
    row_exact_keys = list(row_exact_qmv._plans(row_exact_m.attn.q_proj).keys())
    assert any(
        k[0] == 3 for k in row_exact_keys
    ), f"row-exact opt-in model must still plan group3: {row_exact_keys}"

    # --- 4. pre-a1-529daae8 Mutant 2 (unaligned N % 8 != 0 ref_proof_tasks) ---
    monkeypatch.setattr(kernels, "_PROOFS", {})
    monkeypatch.setattr(row_exact_qmv, "_UNROLLED_VERDICTS", {})

    class _UnalignedAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, _ = _quantized_format_linear(
                fmt, 128, 256, 8, 64, dtype, seed=441
            )
            self.k_proj, _ = _quantized_format_linear(
                fmt, 64, 256, 8, 64, dtype, seed=442
            )
            self.v_proj, _ = _quantized_format_linear(
                fmt, 64, 256, 8, 64, dtype, seed=443
            )
            # N=68 is divisible by 4 (group1 rps=4 -> 2*rps=8? Use N=72? No,
            # 2*rps=8 requires N%8==0 for o_proj; unaligned N%8!=0 hits
            # _plan_warm_target directly with rps=2 / N=68).
            self.o_proj, _ = _quantized_format_linear(
                fmt, 68, 256, 8, 64, dtype, seed=444
            )

    with testing.verified("qwen4_exp"):
        ua = _UnalignedAttn()
    insts: dict = {}
    unrolled: dict = {}
    refs: dict = {}
    row_exact_qmv._plan_warm_target(
        (ua.o_proj,),
        256,
        2,
        dtype,
        (2, 2, True),
        grouped=True,
        instances=insts,
        unrolled_tasks=unrolled,
        ref_proof_tasks=refs,
    )
    o_plan = linear._plan(ua.o_proj)
    assert o_plan is not None and o_plan.n % 8 != 0
    assert (id(o_plan.proof), (dtype, 1)) in refs

    # --- 5. pre-a1-529daae8 Mutant 3 (pre-populated _UNROLLED_VERDICTS[shape]=True
    #        still builds inst into inst.kernel._built) ---
    orig_build = row_exact_qmv.row_exact_skeleton.build_plan
    captured_u: list = []

    def _cap_u(faces, shapes, k, rows, dt, geometry, **kwargs):
        plan = orig_build(faces, shapes, k, rows, dt, geometry, **kwargs)
        if plan is not None and geometry[2]:
            in_specs, _ = row_exact_qmv._plan_stand_in(plan, k, dt)
            inst = kernels.planned(
                plan.kernel, in_specs, plan.output_dtypes, template=plan.template
            )
            inst.kernel._built.clear()
            row_exact_qmv._UNROLLED_VERDICTS[plan.shape] = True
            captured_u.append(inst)
        return plan

    monkeypatch.setattr(row_exact_qmv.row_exact_skeleton, "build_plan", _cap_u)
    for fut in row_exact_qmv.warm_model(disarmed_m, mtp_enabled=True):
        fut.result()
    assert captured_u, "expected unrolled attention instances"
    for inst in captured_u:
        assert (
            inst in inst.kernel._built
        ), "pre-verified unrolled tile must still build into kernel._built"


def test_warm_hc_fused_zero_compiles_and_mutants(monkeypatch):
    """Release/4 U21 hc_fused warming:
    1) Subprocess Metal spy: zero omlx_qwen4_hc_* Metal compiles, zero
       _VALIDATED misses (mid-forward mx.eval syncs), and zero mx.compile calls
       on request #0 across prefill (write=False/True, INJ=1 tail_launch &
       INJ=0 _tail) and decode/batch/verify (rows=1,2,3,4,7,16, write=False/True),
       bitwise-identical to unwarmed outputs;
    2) Concurrent warm_model thread safety + O(1) second-call no-op;
    3) Fail-closed on broken face kernel: raises FaceKernelError and never marks
       launch recorded or validated;
    4) Non-Qwen4 model warm never imports mlx_vlm.models.qwen4_exp.hc_fused.
    """
    import concurrent.futures
    import subprocess
    import sys
    import threading
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
    qwen35_moe_routed_decode,
)

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen4_exp import hc_fused
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedResidual, Qwen4ExpRMSNorm

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
            m = re.search(r"omlx_qwen4_hc_[A-Za-z0-9_]*", text)
            if m:
                with lock:
                    recorded[phase[0]].append(m.group(0))
    return res


objc.method_setImplementation(method, C.cast(hooked, ptr))

HC, HIDDEN, LOWRANK = 4, 512, 128
WIDTH = HC * HIDDEN
dtype = mx.bfloat16
fmt = affine.AffineFormat()


def make_fl(k, n, seed):
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


def make_hc(use_combine, seed):
    mod = Qwen4ExpGatedResidual.__new__(Qwen4ExpGatedResidual)
    nn.Module.__init__(mod)
    mod.hc_count, mod.hidden_size, mod.hc_lowrank = HC, HIDDEN, LOWRANK
    mod.hc_norm = Qwen4ExpRMSNorm(WIDTH, group_size=HIDDEN, eps=1e-6)
    mx.random.seed(seed)
    mod.hc_norm.weight = (mx.random.normal((WIDTH,)) * 0.05).astype(dtype)
    mod.input_mix_weight_down = make_fl(WIDTH, LOWRANK, seed + 1)
    mod.input_mix_weight_up = make_fl(LOWRANK, WIDTH, seed + 2)
    if use_combine:
        mod.block_inject_weight = make_fl(WIDTH, HC, seed + 3)
    mod.eval()
    mx.eval(mod.parameters())
    return mod


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hc_inj0 = make_hc(True, 10)
        self.hc_inj1 = make_hc(True, 20)
        self.hc_mix = make_hc(False, 30)


with testing.verified("qwen4_exp"):
    model = _Model()
    hc_fused.rebuild_plans(model)

    # Compute unwarmed reference outputs on a separate cloned model? Even simpler:
    # run reference after resetting hc_fused caches or compute eager reference now?
    # Keep subprocess clean: record bit-exact outputs via warmed vs cleared cache.
    phase[0] = "load"
    for fut in linear.warm_existing_dense(model, dtypes=(dtype,)) or ():
        fut.result()
    qwen35_moe_routed_decode.warm_moe_model(
        model, mtp_enabled=True, warm_dispatch=False
    )
    qwen35_moe_routed_decode.warm_engine_dispatch(
        model,
        mx.default_stream(mx.gpu),
        prefill_lengths=(32,),
        mtp_enabled=True,
    )

# Prepare inputs before switching to request0 phase.
mx.random.seed(99)
inputs = {}
for rows in (1, 2, 3, 4, 7, 16, 32):
    x = (mx.random.normal((1, rows, WIDTH)) * 0.1).astype(dtype)
    br = (mx.random.normal((1, rows, HIDDEN)) * 0.1).astype(dtype)
    gt = (mx.random.normal((1, rows, HC)) * 0.1).astype(dtype)
    mx.eval(x, br, gt)
    inputs[rows] = (x, br, gt)

# Spy mid-forward mx.eval syncs (which happen ONLY on _VALIDATED misses) and
# mx.compile calls during request0.
mid_evals = []
compile_calls = []
orig_eval = mx.eval
orig_compile = mx.compile


def spy_eval(*args, **kwargs):
    mid_evals.append(len(args))
    return orig_eval(*args, **kwargs)


def spy_compile(*args, **kwargs):
    compile_calls.append(1)
    return orig_compile(*args, **kwargs)


phase[0] = "request0"
mx.eval = spy_eval
mx.compile = spy_compile
try:
    warm_outs = []
    # Prefill (rows=32) on both inject (tail_launch) and mixer (_tail)
    x32, br32, gt32 = inputs[32]
    for mod in (model.hc_inj0, model.hc_inj1, model.hc_mix):
        warm_outs.append(hc_fused.prefill_forward(mod, x32, write=None, nax=False))
        warm_outs.append(hc_fused.prefill_forward(mod, x32, write=(br32, gt32), nax=False))
    # Decode / batch / verify (rows=1,2,3,4,7,16) on inject & mixer
    for rows in (1, 2, 3, 4, 7, 16):
        xr, brr, gtr = inputs[rows]
        for mod in (model.hc_inj0, model.hc_inj1, model.hc_mix):
            warm_outs.append(hc_fused.fused_forward(mod, xr, write=None))
            warm_outs.append(hc_fused.fused_forward(mod, xr, write=(brr, gtr)))
finally:
    mx.eval = orig_eval
    mx.compile = orig_compile

flat_warm = []
for item in warm_outs:
    if isinstance(item, tuple):
        flat_warm.extend(item)
    else:
        flat_warm.append(item)
mx.eval(*flat_warm)

# Bitwise parity against cold validation path (clear _VALIDATED & recorded).
hc_fused._VALIDATED.clear()
cold_outs = []
x32, br32, gt32 = inputs[32]
for mod in (model.hc_inj0, model.hc_inj1, model.hc_mix):
    cold_outs.append(hc_fused.prefill_forward(mod, x32, write=None, nax=False))
    cold_outs.append(hc_fused.prefill_forward(mod, x32, write=(br32, gt32), nax=False))
for rows in (1, 2, 3, 4, 7, 16):
    xr, brr, gtr = inputs[rows]
    for mod in (model.hc_inj0, model.hc_inj1, model.hc_mix):
        cold_outs.append(hc_fused.fused_forward(mod, xr, write=None))
        cold_outs.append(hc_fused.fused_forward(mod, xr, write=(brr, gtr)))
flat_cold = []
for item in cold_outs:
    if isinstance(item, tuple):
        flat_cold.extend(item)
    else:
        flat_cold.append(item)
mx.eval(*flat_cold)

bitwise_ok = len(flat_warm) == len(flat_cold) and all(
    mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()
    for a, b in zip(flat_warm, flat_cold)
)

print(
    json.dumps(
        {
            "load_count": len(recorded["load"]),
            "request0": recorded["request0"],
            "mid_evals": mid_evals,
            "compile_calls": compile_calls,
            "bitwise_ok": bitwise_ok,
        }
    )
)
"""
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["load_count"] > 0, f"expected hc_fused Metal warms at load: {data}"
    assert (
        data["request0"] == []
    ), f"hc_fused Metal compiled on request #0: {data['request0']}"
    assert (
        data["mid_evals"] == []
    ), f"mid-forward mx.eval syncs on request #0: {data['mid_evals']}"
    assert (
        data["compile_calls"] == []
    ), f"mx.compile calls on request #0: {data['compile_calls']}"
    assert data["bitwise_ok"] is True

    # Part 2: Thread safety + O(1) second-call no-op + fail-closed on broken build.
    from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import hc_fused
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedResidual, Qwen4ExpRMSNorm

    fmt = affine.AffineFormat()
    dtype = mx.bfloat16
    hc, hidden, lowrank = 4, 512, 128
    width = hc * hidden

    def _make_hc(seed):
        mod = Qwen4ExpGatedResidual.__new__(Qwen4ExpGatedResidual)
        nn.Module.__init__(mod)
        mod.hc_count, mod.hidden_size, mod.hc_lowrank = hc, hidden, lowrank
        mod.hc_norm = Qwen4ExpRMSNorm(width, group_size=hidden, eps=1e-6)
        mod.hc_norm.weight = mx.zeros((width,), dtype=dtype)
        mod.input_mix_weight_down, _ = _quantized_format_linear(
            fmt, lowrank, width, 8, 64, dtype, seed=seed
        )
        mod.input_mix_weight_up, _ = _quantized_format_linear(
            fmt, width, lowrank, 8, 64, dtype, seed=seed + 1
        )
        mod.block_inject_weight, _ = _quantized_format_linear(
            fmt, hc, width, 8, 64, dtype, seed=seed + 2
        )
        mod.eval()
        mx.eval(mod.parameters())
        return mod

    monkeypatch.setattr(hc_fused, "_VALIDATED", set())
    monkeypatch.setattr(hc_fused, "_KERNELS", {})
    monkeypatch.setattr(kernels, "_FACE_KERNELS", {})
    monkeypatch.setattr(kernels, "_RAW_KERNELS", {})
    monkeypatch.setattr(hc_fused, "_decode_launch", hc_fused._decode_launch.__wrapped__)
    with testing.verified("qwen4_exp"):
        hc_mod = _make_hc(501)
        hc_fused.rebuild_plans(hc_mod)

    barrier = threading.Barrier(2)

    def _run_concurrent():
        barrier.wait()
        futs = hc_fused.warm_model(hc_mod)
        for fut in futs:
            fut.result()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(_run_concurrent)
        f2 = pool.submit(_run_concurrent)
        f1.result()
        f2.result()

    assert hc_fused.warm_model(hc_mod) == ()
    plan = hc_mod.__dict__[hc_fused._PLAN_ATTR]
    assert plan.prefill_signatures[False] in hc_fused._VALIDATED
    assert plan.prefill_signatures[True] in hc_fused._VALIDATED
    for row_launches in plan.decode_launches:
        assert row_launches[False] in hc_fused._VALIDATED
        assert row_launches[True] in hc_fused._VALIDATED

    # Part 3: Fail-closed on broken build: raises FaceKernelError and leaves
    # _VALIDATED untouched.
    monkeypatch.setattr(hc_fused, "_VALIDATED", set())
    monkeypatch.setattr(hc_fused, "_KERNELS", {})
    monkeypatch.setattr(kernels, "_FACE_KERNELS", {})
    monkeypatch.setattr(kernels, "_RAW_KERNELS", {})
    with testing.verified("qwen4_exp"):
        bad_mod = _make_hc(511)
        hc_fused.rebuild_plans(bad_mod)

    def _boom(*_a, **_kw):
        raise RuntimeError("injected metal compile failure")

    monkeypatch.setattr(kernels._Instance, "build", _boom)
    bad_futs = hc_fused.warm_model(bad_mod)
    assert bad_futs, "expected unbuilt warm futures"
    with pytest.raises(
        hc_fused.FaceKernelError, match="injected metal compile failure"
    ):
        for fut in bad_futs:
            fut.result()
    assert len(hc_fused._VALIDATED) == 0

    # Part 4: Non-Qwen4 model warm never imports mlx_vlm.models.qwen4_exp.hc_fused.
    non_qwen = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, mlx.nn as nn\n"
                "from omlx.patches import qwen35_moe_routed_decode\n"
                "class M(nn.Module):\n"
                "    pass\n"
                "m = M()\n"
                "qwen35_moe_routed_decode.warm_moe_model(m, warm_dispatch=True)\n"
                "qwen35_moe_routed_decode.warm_engine_dispatch(m, None)\n"
                "assert 'mlx_vlm.models.qwen4_exp.hc_fused' not in sys.modules\n"
            ),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert non_qwen.returncode == 0, non_qwen.stderr


def test_warm_engine_dispatch_joins_row_futures_when_hc_warm_raises(monkeypatch):
    """pre-a1-a69f8379 LOW 1 + all 5 surviving mutants:
    1) warm_engine_dispatch joins row_exact_qmv futures if hc_fused.warm_model raises;
    2) linear.py:1476 mutant (mtp_enabled=False still warms Qwen GDN stacked (1, 2, K));
    3) hc_fused.py:2271 mutant (write_enabled()==False skips write=True warms);
    4) hc_fused.py:2304/2404 mutant (FP32 HC plan skips prefill norms, _act, _tail);
    5) hc_fused.py:2399 mutant (prefill_lengths filters out bool and <= MAX_ROWS);
    6) qwen35_moe_routed_decode.py:1896 mutant (vendor-path sys.modules fallback).
    """
    import sys
    import types

    from omlx.formats import tile_dense
    from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
    from omlx.patches import qwen35_gdn_prework, row_exact_qmv

    if not mx.metal.is_available():
        pytest.skip("requires Metal")

    # --- LOW 1: warm_engine_dispatch joins submitted row_exact_qmv futures ---
    spy_fut = _Future()

    class _EmptyModel(nn.Module):
        pass

    monkeypatch.setattr(
        row_exact_qmv, "warm_model", lambda _m, mtp_enabled=False: (spy_fut,)
    )
    fake_hc = types.SimpleNamespace(
        warm_model=lambda _m: (_ for _ in ()).throw(RuntimeError("hc boom"))
    )
    monkeypatch.setattr(qwen35_moe_routed_decode, "_hc_fused_module", lambda: fake_hc)
    with pytest.raises(RuntimeError, match="hc boom"):
        qwen35_moe_routed_decode.warm_engine_dispatch(
            _EmptyModel(), None, mtp_enabled=True
        )
    assert spy_fut.awaited == 1
    monkeypatch.undo()

    # --- Mutant 1: mtp_enabled=False warms Qwen GDN stacked (1, 2, 2560) ---
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import hc_fused
    from mlx_vlm.models.qwen4_exp.language import (
        Qwen4ExpGatedDeltaNet,
        Qwen4ExpGatedResidual,
        Qwen4ExpRMSNorm,
        Qwen4ExpRMSNormGated,
    )

    fmt = affine.AffineFormat()
    dtype = mx.bfloat16
    hk, hv, dk, dv = 16, 48, 128, 128
    c_dim = 2 * hk * dk + hv * dv

    def _make_qwen_fl(n, k, seed):
        layer, _ = _quantized_format_linear(fmt, n, k, 8, 64, dtype, seed=seed)
        return layer

    def _make_gdn_model():
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
        mod.in_proj_qkv = _make_qwen_fl(c_dim, 2560, 611)
        mod.in_proj_z = _make_qwen_fl(hv * dv, 2560, 612)
        mod.in_proj_b = _make_qwen_fl(hv, 2560, 613)
        mod.in_proj_a = _make_qwen_fl(hv, 2560, 614)
        mod.out_proj = _make_qwen_fl(2560, hv * dv, 615)
        mod.eval()
        mx.eval(mod.parameters())

        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.gdn = mod

        return _M()

    monkeypatch.setattr(kernels, "_PROOFS", {})
    with testing.verified("qwen4_exp"):
        gdn_m = _make_gdn_model()
        for fut in linear.warm_existing_dense(gdn_m, dtypes=(dtype,)) or ():
            fut.result()
    qwen35_moe_routed_decode.warm_moe_model(
        gdn_m, mtp_enabled=False, warm_dispatch=False
    )
    qwen35_moe_routed_decode.warm_engine_dispatch(
        gdn_m, mx.default_stream(mx.gpu), prefill_lengths=(), mtp_enabled=False
    )
    dec_plan = qwen35_gdn_prework._qwen4_decode_plan(gdn_m.gdn)
    assert dec_plan is not None
    stacked = dec_plan[-1]
    cold = []
    orig_keep = linear._FormatLayer._keep
    orig_bp = tile_dense.build_plan
    orig_compile = mx.compile
    orig_rf = linear.FormatLinear._register_fast
    orig_rdf = linear.FormatLinear._register_dot_fast
    monkeypatch.setattr(
        linear._FormatLayer,
        "_keep",
        lambda self: (cold.append("keep") or orig_keep(self)),
    )
    monkeypatch.setattr(
        tile_dense,
        "build_plan",
        lambda *a, **kw: (cold.append("bp") or orig_bp(*a, **kw)),
    )
    monkeypatch.setattr(
        mx,
        "compile",
        lambda *a, **kw: (cold.append("compile") or orig_compile(*a, **kw)),
    )
    monkeypatch.setattr(
        linear.FormatLinear,
        "_register_fast",
        lambda self, *a, **kw: (cold.append("rf") or orig_rf(self, *a, **kw)),
    )
    monkeypatch.setattr(
        linear.FormatLinear,
        "_register_dot_fast",
        lambda self, *a, **kw: (cold.append("rdf") or orig_rdf(self, *a, **kw)),
    )
    p0 = kernels.proof_count()
    y_b2 = stacked(mx.zeros((1, 2, 2560), dtype=dtype))
    mx.eval(y_b2)
    assert kernels.proof_count() == p0
    assert (
        cold == []
    ), f"GDN stacked (1, 2, 2560) did cold work with mtp_enabled=False: {cold}"
    monkeypatch.undo()

    # Helper to build a small Qwen4ExpGatedResidual for mutants 2-4.
    def _make_hc(dt, use_combine, seed):
        hc, hidden, lowrank = 4, 512, 128
        width = hc * hidden
        mod = Qwen4ExpGatedResidual.__new__(Qwen4ExpGatedResidual)
        nn.Module.__init__(mod)
        mod.hc_count, mod.hidden_size, mod.hc_lowrank = hc, hidden, lowrank
        mod.hc_norm = Qwen4ExpRMSNorm(width, group_size=hidden, eps=1e-6)
        mod.hc_norm.weight = mx.zeros((width,), dtype=dt)
        mod.input_mix_weight_down, _ = _quantized_format_linear(
            fmt, lowrank, width, 8, 64, dt, seed=seed
        )
        mod.input_mix_weight_up, _ = _quantized_format_linear(
            fmt, width, lowrank, 8, 64, dt, seed=seed + 1
        )
        if use_combine:
            mod.block_inject_weight, _ = _quantized_format_linear(
                fmt, hc, width, 8, 64, dt, seed=seed + 2
            )
        mod.eval()
        mx.eval(mod.parameters())
        return mod

    # --- Mutant 2: write_enabled() == False skips write=True warms ---
    monkeypatch.setattr(hc_fused, "_WRITE_DISABLED", True)
    monkeypatch.setattr(hc_fused, "_VALIDATED", set())
    monkeypatch.setattr(hc_fused, "_KERNELS", {})
    monkeypatch.setattr(kernels, "_FACE_KERNELS", {})
    monkeypatch.setattr(kernels, "_RAW_KERNELS", {})
    monkeypatch.setattr(hc_fused, "_decode_launch", hc_fused._decode_launch.__wrapped__)
    with testing.verified("qwen4_exp"):
        hc_nowrite = _make_hc(mx.bfloat16, True, 620)
        hc_fused.rebuild_plans(hc_nowrite)
    for fut in hc_fused.warm_model(hc_nowrite):
        fut.result()
    plan_nw = hc_nowrite.__dict__[hc_fused._PLAN_ATTR]
    assert plan_nw.prefill_signatures[False] in hc_fused._VALIDATED
    assert plan_nw.prefill_signatures[True] not in hc_fused._VALIDATED
    for row_launches in plan_nw.decode_launches:
        assert row_launches[False] in hc_fused._VALIDATED
        assert row_launches[True] not in hc_fused._VALIDATED
    monkeypatch.undo()

    # --- Mutant 3: FP32 HC plan skips prefill norms, _act, _tail ---
    monkeypatch.setattr(hc_fused, "_VALIDATED", set())
    monkeypatch.setattr(hc_fused, "_KERNELS", {})
    monkeypatch.setattr(kernels, "_FACE_KERNELS", {})
    monkeypatch.setattr(kernels, "_RAW_KERNELS", {})
    monkeypatch.setattr(hc_fused, "_decode_launch", hc_fused._decode_launch.__wrapped__)
    with testing.verified("qwen4_exp"):
        hc_fp32 = _make_hc(mx.float32, False, 630)
        hc_fused.rebuild_plans(hc_fp32)
    for fut in hc_fused.warm_model(hc_fp32):
        fut.result()
    plan_32 = hc_fp32.__dict__[hc_fused._PLAN_ATTR]
    assert plan_32.dtype == mx.float32
    assert plan_32.prefill_signatures[False] not in hc_fused._VALIDATED
    assert plan_32.prefill_signatures[True] not in hc_fused._VALIDATED
    assert "omlx_qwen4_hc_prefill_norm" not in hc_fused._KERNELS
    act_calls = []
    tail_calls = []
    monkeypatch.setattr(
        hc_fused, "_act", lambda hc: (act_calls.append(hc) or (lambda y: y))
    )
    monkeypatch.setattr(
        hc_fused,
        "_tail",
        lambda hc, h: (tail_calls.append((hc, h)) or (lambda u, n: u)),
    )
    hc_fused.warm_model_dispatch(hc_fp32, prefill_lengths=(32,))
    assert act_calls == [] and tail_calls == []
    monkeypatch.undo()

    # --- Mutant 4: prefill_lengths filters out bool and <= MAX_ROWS ---
    with testing.verified("qwen4_exp"):
        hc_mix = _make_hc(mx.bfloat16, False, 640)
        hc_fused.rebuild_plans(hc_mix)
    seen_tail_shapes = []
    orig_tail = hc_fused._tail

    def _spy_tail(hc, h):
        fn = orig_tail(hc, h)

        def _wrapped(u, n):
            seen_tail_shapes.append(u.shape)
            return fn(u, n)

        return _wrapped

    class _IntSub(int):
        pass

    monkeypatch.setattr(hc_fused, "_tail", _spy_tail)
    hc_fused.warm_model_dispatch(
        hc_mix, prefill_lengths=(True, _IntSub(33), 1, hc_fused.MAX_ROWS, 32)
    )
    assert seen_tail_shapes == [(1, 32, 4 * 512)]
    monkeypatch.undo()

    # --- Mutant 5: _hc_fused_module falls back to full vendor sys.modules key ---
    vendor_key = (
        "omlx.patches.mlx_vlm_qwen4_exp_compat.vendor.mlx_vlm.models.qwen4_exp.hc_fused"
    )
    alias_key = "mlx_vlm.models.qwen4_exp.hc_fused"
    sentinel = object()
    monkeypatch.delitem(sys.modules, alias_key, raising=False)
    monkeypatch.setitem(sys.modules, vendor_key, sentinel)
    assert qwen35_moe_routed_decode._hc_fused_module() is sentinel


def test_vfast_c3_schedule_warms_all_routes_and_k_mod_8_zero_cold_work(monkeypatch):
    """VFAST c3 load-time warm-up: mma, msg, wide, mlx, dot, steel/nax, and
    k % 32 != 0 (k % 8 == 0) all warm at load with zero cold proofs, plans,
    compiles, or fast registrations on request #0, and warmed mma/msg graphs
    hold no DensePlan closure."""
    import gc
    import weakref

    from omlx import formats
    from omlx.formats import tile_dense, tuning

    if not mx.metal.is_available():
        pytest.skip("requires Metal")

    fmt = affine.AffineFormat()
    dtype = mx.bfloat16
    dot_pol = tuning.DenseTilePolicy("dot")
    msg_pol = tuning.MsgTilePolicy(4)
    mma_pol = tuning.MmaTilePolicy(16, 32, 8)
    wide_pol = tuning.DenseTilePolicy("wide")
    mlx_pol = tuning.DenseTilePolicy("mlx")

    def _sched_lookup(route, *a, rows=1, **kw):
        if route == "dense_tile" and rows == 2:
            return dot_pol
        if route == "msg_tile" and 3 <= rows <= 4:
            return msg_pol
        if route == "mma_tile" and 5 <= rows <= 8:
            return mma_pol
        if route == "dense_tile" and 9 <= rows <= 12:
            return wide_pol
        if route == "dense_tile" and 13 <= rows <= 16:
            return mlx_pol
        return None

    monkeypatch.setattr(kernels, "_PROOFS", {})
    monkeypatch.setattr(tuning, "lookup", _sched_lookup)
    monkeypatch.setattr(tuning, "row_bounds", lambda route, **kw: (2, 4, 8, 12, 16))

    class _AllRoutesModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.full, _ = _quantized_format_linear(
                fmt, 128, 512, 8, 64, dtype, seed=701
            )

    with testing.verified("qwen4_exp"):
        model = _AllRoutesModel()
        for fut in linear.warm_existing_dense(model, dtypes=(dtype,)) or ():
            fut.result()

    linear.warm_model_dispatch(
        model,
        prefill_lengths=(64,),
        decode_batch_sizes=(1, 2, 4, 8),
        mtp_enabled=True,
        verify_rows=(2, 3, 4, 8, 12, 16),
    )

    kept = model.full.__dict__[linear._KEPT]
    assert tile_dense.route_for(kept.dense, 2, dtype).backend == "dot"
    assert tile_dense.route_for(kept.dense, 4, dtype).backend == "msg"
    assert tile_dense.route_for(kept.dense, 8, dtype).backend == "mma"
    assert tile_dense.route_for(kept.dense, 12, dtype).backend == "wide"
    assert tile_dense.route_for(kept.dense, 16, dtype).backend == "mlx"
    assert kept.dense_fast is not None and kept.dot_fast is not None
    assert {model.full._fast, model.full._fast2} == {kept.dense_fast, kept.dot_fast}

    # Spy all cold work during request #0 across every route.
    cold: list[str] = []
    orig_keep = linear._FormatLayer._keep
    orig_bp = tile_dense.build_plan
    orig_compile = mx.compile
    orig_rf = linear.FormatLinear._register_fast
    orig_rdf = linear.FormatLinear._register_dot_fast
    with monkeypatch.context() as spy:
        spy.setattr(
            linear._FormatLayer,
            "_keep",
            lambda self: (cold.append("keep") or orig_keep(self)),
        )
        spy.setattr(
            tile_dense,
            "build_plan",
            lambda *a, **kw: (cold.append("build_plan") or orig_bp(*a, **kw)),
        )
        spy.setattr(
            mx,
            "compile",
            lambda *a, **kw: (cold.append("compile") or orig_compile(*a, **kw)),
        )
        spy.setattr(
            linear.FormatLinear,
            "_register_fast",
            lambda self, *a, **kw: (
                cold.append("register_fast") or orig_rf(self, *a, **kw)
            ),
        )
        spy.setattr(
            linear.FormatLinear,
            "_register_dot_fast",
            lambda self, *a, **kw: (
                cold.append("register_dot_fast") or orig_rdf(self, *a, **kw)
            ),
        )
        p0 = kernels.proof_count()
        xs = {
            "1": mx.random.normal((1, 1, 512), dtype=dtype),
            "b2": mx.random.normal((2, 1, 512), dtype=dtype),
            "s2": mx.random.normal((1, 2, 512), dtype=dtype),
            "s4": mx.random.normal((1, 4, 512), dtype=dtype),
            "s8": mx.random.normal((1, 8, 512), dtype=dtype),
            "s12": mx.random.normal((1, 12, 512), dtype=dtype),
            "s16": mx.random.normal((1, 16, 512), dtype=dtype),
            "p64": mx.random.normal((1, 64, 512), dtype=dtype),
        }
        mx.eval(*xs.values())
        warm_outs = {k: model.full(v) for k, v in xs.items()}
        mx.eval(*warm_outs.values())
        assert kernels.proof_count() == p0
        assert cold == [], f"unexpected cold work on c3 warmed routes: {cold}"

    # Bitwise parity vs cold un-warmed execution.
    ref_binding = weakref.ref(kept.dense.binding)
    del kept
    monkeypatch.setattr(kernels, "_PROOFS", {})
    model.full.__dict__.pop(linear._KEPT, None)
    model.full.__dict__.pop(linear._FAST, None)
    model.full.__dict__.pop(linear._FAST2, None)
    cold_outs = {k: model.full(v) for k, v in xs.items()}
    mx.eval(*cold_outs.values())
    for tag in xs:
        assert mx.array_equal(
            warm_outs[tag].view(mx.uint16), cold_outs[tag].view(mx.uint16)
        ).item(), f"warm vs cold mismatch on {tag}"

    # Warmed mma/msg graphs in _PROOFS must not pin DensePlan.binding on unload.
    del model, warm_outs, cold_outs, xs
    gc.collect()
    assert ref_binding() is None, "warmed mma/msg graph pinned DensePlan.binding"

    # Non-multiple-of-32 K (k=40, k % 8 == 0, k % 32 != 0) with a tile face
    # warms wide at load and runs request #0 with zero cold compiles/plans.
    from tests.test_formats_tile_dense import _stack, _tile_toy

    monkeypatch.setattr(kernels, "_PROOFS", {})
    monkeypatch.setattr(
        tuning,
        "lookup",
        lambda route, *a, rows=1, **kw: (
            wide_pol if route == "dense_tile" and 2 <= rows <= 16 else None
        ),
    )
    monkeypatch.setattr(tuning, "row_bounds", lambda route, **kw: (4, 16))
    face_k40 = _tile_toy("warm_k40_wide")

    class _K40Format(formats.QuantFormat):
        mode = "test_warm_k40"
        codes_name = "codes"
        tensors = ("group_scales",)

        def layout(self, k, n, bits, group_size):
            return {"codes": ((n, k), mx.uint8), "group_scales": ((n,), mx.float32)}

        def decode(self, tensors, bits, group_size):
            return (tensors["codes"].astype(mx.float32) - 128.0) * tensors[
                "group_scales"
            ][:, None]

        def tile(self, layer, k, n):
            return face_k40

    k40_weights = {name: a[0] for name, a in _stack(32, 40, seed=71).items()}

    class _K40Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = FormatLinear(_K40Format(), 8, 1, 40, k40_weights)

    with testing.verified("qwen4_exp"):
        m40 = _K40Model()
        for fut in linear.warm_existing_dense(m40, dtypes=(dtype,)) or ():
            fut.result()
    linear.warm_model_dispatch(
        m40, prefill_lengths=(), decode_batch_sizes=(1, 4), mtp_enabled=False
    )
    p40 = kernels.proof_count()
    cold40: list[str] = []
    with monkeypatch.context() as spy:
        spy.setattr(
            tile_dense,
            "build_plan",
            lambda *a, **kw: (cold40.append("bp") or orig_bp(*a, **kw)),
        )
        spy.setattr(
            mx,
            "compile",
            lambda *a, **kw: (cold40.append("compile") or orig_compile(*a, **kw)),
        )
        y40 = m40.proj(mx.zeros((4, 1, 40), dtype=dtype))
        mx.eval(y40)
    assert kernels.proof_count() == p40
    assert cold40 == [], f"k=40 wide route did cold work: {cold40}"


def test_warm_moe_model_warms_batched_and_coalesced_verify_zero_proofs(
    monkeypatch, tmp_path
):
    """When per-chip tuning enables single-stream coalescing for M=4,
    warm_moe_model(mtp_enabled=True) proves both the uncoalesced batched
    decode Metal (4, 1, H) and the coalesced single-stream verify Metal
    (1, 4, H), so serving both triggers 0 new proofs."""
    if not mx.metal.is_available():
        pytest.skip("requires Metal")

    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP
    from mlx_vlm.models.switch_layers import SwiGLU
    from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

    from omlx.formats import tuning
    from tests.test_formats_tables import tabled_dot

    monkeypatch.setattr(kernels, "_PROOFS", {})
    monkeypatch.setattr(kernels, "_RAW_KERNELS", {})
    tabled_dot(monkeypatch)

    hidden, inter, experts, top_k = 2560, 640, 256, 10
    dtype = mx.bfloat16
    chip = "applegpu_g17s_c40_warmtest"
    v1_dir = tmp_path / "v1"
    v1_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tuning, "_TABLE_DIRS", [v1_dir])
    monkeypatch.setattr(tuning, "_EXTRA_DIRS", [])
    monkeypatch.setattr(tuning, "current_chip", lambda: chip)
    if hasattr(qwen35_moe_routed_decode, "_ensure_tuning_registered"):
        qwen35_moe_routed_decode._ensure_tuning_registered()
        skel = tuning.skeleton_digest("routed_window")
        (v1_dir / f"{chip}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "chip": chip,
                    "entries": [
                        {
                            "key": {
                                "route": "routed_window",
                                "face_id": "*:lut_small",
                                "dtype": "bf16",
                                "bits": 4,
                                "group": 64,
                                "rows_bucket": "4",
                                "n": 2 * inter,
                                "k": hidden,
                            },
                            "face_sha256": "",
                            "skeleton_sha256": skel,
                            "measured_at": "2026-11-24T20:00:00Z",
                            "exactness": "bitwise",
                            "params": {"coalesce": "unfolded"},
                        }
                    ],
                }
            )
        )
    tuning.reset_cache()
    try:
        fmt = affine.AffineFormat()

        class _FlashMiniBlock(nn.Module):
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
                    fmt, experts, 2 * inter, hidden, 4, 64, dtype, seed=1
                )
                sw.down_proj, _ = _quantized_format_switch(
                    fmt, experts, hidden, inter, 4, 64, dtype, seed=2
                )
                self.switch_mlp = sw
                sh = Qwen3_5MLP.__new__(Qwen3_5MLP)
                nn.Module.__init__(sh)
                sh.gate_proj, _ = _quantized_format_linear(
                    fmt, inter, hidden, 8, 128, dtype, seed=3
                )
                sh.up_proj, _ = _quantized_format_linear(
                    fmt, inter, hidden, 8, 128, dtype, seed=4
                )
                sh.down_proj, _ = _quantized_format_linear(
                    fmt, hidden, inter, 8, 128, dtype, seed=5
                )
                self.shared_expert = sh
                self.shared_expert_gate, _ = _quantized_format_linear(
                    fmt, 1, hidden, 8, 64, dtype, seed=6
                )

        with testing.verified("qwen4_exp"):
            block = _FlashMiniBlock()
            block.eval()

        compiled_ss: dict[tuple[int, ...], bool] = {}
        orig_window = qwen35_moe_routed_decode._window

        def spy_window(launches, folded, *, single_stream=False):
            fn = (
                orig_window(launches, folded, single_stream=single_stream)
                if hasattr(qwen35_moe_routed_decode, "_coalesce_mode")
                else orig_window(launches, folded)
            )

            def traced(x, operands):
                compiled_ss[tuple(x.shape)] = single_stream
                return fn(x, operands)

            return traced

        monkeypatch.setattr(qwen35_moe_routed_decode, "_window", spy_window)

        qwen35_moe_routed_decode.warm_moe_model(block, mtp_enabled=True)
        before = kernels.proof_count()

        plan = qwen35_moe_routed_decode._cached_plan(block, dtype)
        assert plan is not None and plan.window is not None
        assert (4, experts, False) in plan.window.ran
        assert (4, experts, False, "unfolded") in plan.window.ran
        assert compiled_ss.get((4, 1, hidden)) is False, compiled_ss
        assert compiled_ss.get((1, 4, hidden)) is True, compiled_ss

        y_batch = qwen35_moe_routed_decode.routed_verify_window(
            block, mx.zeros((4, 1, hidden), dtype=dtype)
        )
        y_single = qwen35_moe_routed_decode.routed_verify_window(
            block, mx.zeros((1, 4, hidden), dtype=dtype)
        )
        assert y_batch is not None and y_single is not None
        mx.eval(y_batch, y_single)
        assert kernels.proof_count() == before
    finally:
        tuning.reset_cache()


def test_warm_moe_model_mtp_false_skips_coalesced_and_warms_routed_down(
    monkeypatch, tmp_path
):
    """pre-p3-d32d3543 surviving mutant (:2287) + Candidate 3 warm:
    - mtp_enabled=False with a coalescing table must NOT prove coalesced
      single-stream window metals, while still warming M=4 split routed_down.
    - mtp_enabled=True with a routed_down split table warms M=4 (rps=4) and
      M=8/16 (rps=8) split Metals so serving them triggers 0 new proofs."""
    if not mx.metal.is_available():
        pytest.skip("requires Metal")

    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP
    from mlx_vlm.models.switch_layers import SwiGLU
    from mlx_vlm.models.switch_layers import SwitchGLU as VLMSwitchGLU

    from omlx.formats import tuning
    from tests.test_formats_tables import tabled_dot

    monkeypatch.setattr(kernels, "_PROOFS", {})
    monkeypatch.setattr(kernels, "_RAW_KERNELS", {})
    tabled_dot(monkeypatch)

    hidden, inter, experts, top_k = 2560, 640, 256, 10
    dtype = mx.bfloat16
    chip = "applegpu_g17s_c40_warm_c3"
    v1_dir = tmp_path / "v1"
    v1_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tuning, "_TABLE_DIRS", [v1_dir])
    monkeypatch.setattr(tuning, "_EXTRA_DIRS", [])
    monkeypatch.setattr(tuning, "current_chip", lambda: chip)
    qwen35_moe_routed_decode._ensure_tuning_registered()
    rw_skel = tuning.skeleton_digest("routed_window")
    rd_skel = tuning.skeleton_digest("routed_down")
    assert rd_skel is not None
    entries = [
        {
            "key": {
                "route": "routed_window",
                "face_id": "*:lut_small",
                "dtype": "bf16",
                "bits": 4,
                "group": 64,
                "rows_bucket": "4",
                "n": 2 * inter,
                "k": hidden,
            },
            "face_sha256": "",
            "skeleton_sha256": rw_skel,
            "measured_at": "2026-11-25T01:15:00Z",
            "exactness": "bitwise",
            "params": {"coalesce": "unfolded"},
        },
        {
            "key": {
                "route": "routed_down",
                "face_id": "*:lut_small",
                "dtype": "bf16",
                "bits": 4,
                "group": 64,
                "rows_bucket": "4",
                "n": hidden,
                "k": inter,
            },
            "face_sha256": "",
            "skeleton_sha256": rd_skel,
            "measured_at": "2026-11-25T01:15:00Z",
            "exactness": "bitwise",
            "params": {"mode": "split", "rps": 4, "unroll_pairs": True},
        },
        {
            "key": {
                "route": "routed_down",
                "face_id": "*:lut_small",
                "dtype": "bf16",
                "bits": 4,
                "group": 64,
                "rows_bucket": "6-8",
                "n": hidden,
                "k": inter,
            },
            "face_sha256": "",
            "skeleton_sha256": rd_skel,
            "measured_at": "2026-11-25T01:15:00Z",
            "exactness": "bitwise",
            "params": {"mode": "split", "rps": 8, "unroll_pairs": True},
        },
    ]
    (v1_dir / f"{chip}.json").write_text(
        json.dumps({"schema_version": 1, "chip": chip, "entries": entries})
    )
    tuning.reset_cache()
    try:
        fmt = affine.AffineFormat()

        class _FlashC3Block(nn.Module):
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
                    fmt, experts, 2 * inter, hidden, 4, 64, dtype, seed=11
                )
                sw.down_proj, _ = _quantized_format_switch(
                    fmt, experts, hidden, inter, 4, 64, dtype, seed=12
                )
                self.switch_mlp = sw
                sh = Qwen3_5MLP.__new__(Qwen3_5MLP)
                nn.Module.__init__(sh)
                sh.gate_proj, _ = _quantized_format_linear(
                    fmt, inter, hidden, 8, 128, dtype, seed=13
                )
                sh.up_proj, _ = _quantized_format_linear(
                    fmt, inter, hidden, 8, 128, dtype, seed=14
                )
                sh.down_proj, _ = _quantized_format_linear(
                    fmt, hidden, inter, 8, 128, dtype, seed=15
                )
                self.shared_expert = sh
                self.shared_expert_gate, _ = _quantized_format_linear(
                    fmt, 1, hidden, 8, 64, dtype, seed=16
                )

        with testing.verified("qwen4_exp"):
            block = _FlashC3Block()
            block.eval()

        # mtp_enabled=False: warms batched M=4 split, skips coalesced single-stream.
        qwen35_moe_routed_decode.warm_moe_model(block, mtp_enabled=False)
        plan = qwen35_moe_routed_decode._cached_plan(block, dtype)
        assert plan is not None and plan.window is not None
        assert (4, experts, False) in plan.window.ran
        assert (4, experts, False, "unfolded") not in plan.window.ran
        before_off = kernels.proof_count()
        y4 = qwen35_moe_routed_decode.routed_verify_window(
            block, mx.zeros((4, 1, hidden), dtype=dtype)
        )
        assert y4 is not None
        mx.eval(y4)
        assert kernels.proof_count() == before_off

        # mtp_enabled=True: also warms M=8 split rps=8 -> zero new proofs at (4, 2, H).
        qwen35_moe_routed_decode.warm_moe_model(block, mtp_enabled=True)
        assert (8, experts, False) in plan.window.ran
        before_mtp = kernels.proof_count()
        y8 = qwen35_moe_routed_decode.routed_verify_window(
            block, mx.zeros((4, 2, hidden), dtype=dtype)
        )
        assert y8 is not None
        mx.eval(y8)
        assert kernels.proof_count() == before_mtp
    finally:
        tuning.reset_cache()
