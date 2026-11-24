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
