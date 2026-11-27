# SPDX-License-Identifier: Apache-2.0
"""Fused hyper-connection kernels: parity with the canonical path, eligibility, kill switch."""

from __future__ import annotations

import dataclasses
from unittest.mock import Mock

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from omlx.custom_kernels.nax import is_nax_available
from omlx.formats import Table
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from tests.test_mlx_vlm_qwen4_exp_compat import _tiny_config


@pytest.fixture(autouse=True)
def _vendored_qwen4():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()


HC, HIDDEN, LOWRANK = 4, 2560, 320
WIDTH = HC * HIDDEN


def _affine_format_linear(k, n, bits, group_size, dtype):
    """The matrix the loader builds, a built-in affine FormatLinear, with
    checkpoint-like statistics: positive scales, small biases."""
    from omlx.formats import linear as formats_linear
    from omlx.formats import testing

    projection = nn.QuantizedLinear(k, n, bias=False, group_size=group_size, bits=bits)
    scales = (mx.abs(mx.random.normal(projection.scales.shape)) * 0.01 + 0.002).astype(
        dtype
    )
    biases = (mx.random.normal(projection.biases.shape) * 0.005).astype(dtype)
    with testing.verified("qwen4_exp"):
        return formats_linear.FormatLinear(
            "affine",
            bits,
            group_size,
            k,
            {"weight": projection.weight, "scales": scales, "biases": biases},
        )


def _module(
    bits: int,
    use_combine: bool = True,
    hidden: int = HIDDEN,
    group_size: int = 64,
    dtype=mx.bfloat16,
    lowrank: int = LOWRANK,
):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedResidual, Qwen4ExpRMSNorm

    width = HC * hidden
    module = Qwen4ExpGatedResidual.__new__(Qwen4ExpGatedResidual)
    nn.Module.__init__(module)
    module.hc_count, module.hidden_size, module.hc_lowrank = HC, hidden, lowrank
    module.hc_norm = Qwen4ExpRMSNorm(width, group_size=hidden, eps=1e-6)
    module.hc_norm.weight = (mx.random.normal((width,)) * 0.05).astype(dtype)
    module.input_mix_weight_down = _affine_format_linear(
        width, lowrank, bits, group_size, dtype
    )
    module.input_mix_weight_up = _affine_format_linear(
        lowrank, width, bits, group_size, dtype
    )
    if use_combine:
        module.block_inject_weight = _affine_format_linear(
            width, HC, bits, group_size, dtype
        )
    mx.eval(module.parameters())
    return module


def _format_hc_layers(model):
    """The loader builds FormatLinear; a hand-quantized fixture converts its
    hyper-connection children so the fused route sees what a load produces."""
    from omlx.formats import linear as formats_linear
    from omlx.formats import testing

    targets = [
        (path, module)
        for path, module in model.named_modules()
        if isinstance(module, nn.QuantizedLinear) and "hyper_connection" in path
    ]
    for path, module in targets:
        with testing.verified("qwen4_exp"):
            layer = formats_linear.FormatLinear(
                "affine",
                module.bits,
                module.group_size,
                module.weight.shape[-1] * 32 // module.bits,
                {name: module[name] for name in ("weight", "scales", "biases")},
            )
        parent = model
        *head, leaf = path.split(".")
        for part in head:
            parent = parent[int(part)] if isinstance(parent, list) else parent[part]
        if isinstance(parent, list):
            parent[int(leaf)] = layer
        else:
            parent[leaf] = layer
    return model


def _reference_fp32(module, x):
    def dequant(q):
        return mx.dequantize(
            q.weight, q.scales, q.biases, group_size=q.group_size, bits=q.bits
        ).astype(mx.float32)

    normed = module.hc_norm(x).astype(mx.float32)
    mix = nn.silu((normed @ dequant(module.input_mix_weight_down).T) / HC)
    gate = mx.sigmoid(mix @ dequant(module.input_mix_weight_up).T)
    hidden = module.hidden_size
    mixed = mx.mean(
        gate.reshape(*gate.shape[:-1], HC, hidden)
        * normed.reshape(*normed.shape[:-1], HC, hidden),
        axis=-2,
    )
    if "block_inject_weight" not in module:
        return mixed, None
    return mixed, 2 * mx.sigmoid((normed @ dequant(module.block_inject_weight).T) / HC)


def _rel_l2(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    error = mx.sqrt(mx.sum((a - b) ** 2))
    norm = mx.sqrt(mx.sum(b**2))
    return float((error / norm).item())


def _ulps(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    ulp = float(mx.abs(b).max().item()) * 2.0**-7
    diff = mx.abs(a - b) / ulp
    return float(diff.max().item()), float(diff.mean().item())


# Every bits x layout pair; rows and use_combine rotate so each of their values meets every
# bits value and every layout, and every rows x use_combine pair runs.
@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize(
    "bits,rows,use_combine,group_size,dtype",
    [
        (4, 1, True, 32, mx.bfloat16),
        (4, 4, False, 64, mx.bfloat16),
        (4, 16, True, 64, mx.float16),
        (5, 4, False, 32, mx.bfloat16),
        (5, 16, True, 64, mx.bfloat16),
        (5, 1, False, 64, mx.float16),
        (6, 16, True, 32, mx.bfloat16),
        (6, 1, False, 64, mx.bfloat16),
        (6, 4, True, 64, mx.float16),
        (8, 1, False, 32, mx.bfloat16),
        (8, 4, True, 64, mx.bfloat16),
        (8, 16, False, 64, mx.float16),
    ],
)
def test_fused_matches_canonical_path(bits, rows, use_combine, group_size, dtype):
    mx.random.seed(20260905 + bits * 100 + rows)
    _assert_fused_matches_canonical(
        _module(bits, use_combine, group_size=group_size, dtype=dtype),
        rows,
        use_combine,
    )


# Sizes the checkpoint never has, chosen so every kernel sees a partial final block:
#   768  -> down tail 256 for 4/5-bit;                          norm and inject loops exact
#   1152 -> down tail 128 (all bits), inject tail 128 (4/5-bit), norm tail 128
#   1344 -> down tail 320 (4/5) / 64 (6/8), inject tail 64,       norm tail 64
@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("hidden", [768, 1152, 1344])
def test_fused_matches_canonical_path_at_other_hidden_sizes(hidden, bits, group_size):
    mx.random.seed(20260906 + hidden + bits)
    module = _module(bits, True, hidden=hidden, group_size=group_size)
    _assert_fused_matches_canonical(module, 16, True)


def _assert_fused_matches_canonical(module, rows, use_combine):
    from mlx_vlm.models.qwen4_exp import hc_fused

    hidden = module.hidden_size
    dtype = module.hc_norm.weight.dtype
    x = mx.random.normal((1, rows, HC * hidden)).astype(dtype)
    mx.eval(x)
    assert hc_fused.compatible(module, x)
    fused = hc_fused.fused_forward(module, x)
    assert fused is not None
    canonical = module._forward(x)
    mx.eval(fused, canonical)
    ref_mixed, ref_inject = _reference_fp32(module, x)
    if use_combine:
        fused_mixed, passthrough, fused_inject = fused
        canon_mixed, _, canon_inject = canonical
        assert passthrough is x
        assert fused_inject.shape == canon_inject.shape == (1, rows, HC)
        assert _ulps(fused_inject, canon_inject)[0] <= 4
        assert _ulps(fused_inject, ref_inject)[0] <= 4
    else:
        fused_mixed, canon_mixed = fused, canonical
    assert fused_mixed.shape == canon_mixed.shape == (1, rows, hidden)
    assert fused_mixed.dtype == dtype
    max_vs_canon, mean_vs_canon = _ulps(fused_mixed, canon_mixed)
    assert max_vs_canon <= 16 and mean_vs_canon <= 0.5
    # Both paths round differently; judge each against fp32. The fused path keeps fp32 through
    # the epilogues, so it must stay at least as close to fp32 as the canonical path (with slack).
    max_fused, mean_fused = _ulps(fused_mixed, ref_mixed)
    max_canon, mean_canon = _ulps(canon_mixed, ref_mixed)
    assert max_fused <= max(2 * max_canon, 6)
    assert mean_fused <= mean_canon * 1.5 + 0.05


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("hidden", [HIDDEN, 768, 1152, 1344])
def test_fused_norm_is_bit_identical_to_rms_norm(hidden):
    from mlx_vlm.models.qwen4_exp import hc_fused

    mx.random.seed(7)
    module = _module(4, hidden=hidden)
    width = HC * hidden
    x = (mx.random.normal((1, 4, width)) * 3).astype(mx.bfloat16)
    flat = x.reshape(4, width)
    normed = hc_fused._kernel(
        "omlx_qwen4_hc_fused_norm", ["x", "w", "eps"], ["xn"], hc_fused._N_SOURCE
    )(
        inputs=[flat, module.hc_norm.weight, hc_fused._eps_array(module)],
        template=[("T", mx.bfloat16), ("K", width), ("H", hidden)],
        grid=(256, HC, 4),
        threadgroup=(256, 1, 1),
        output_shapes=[(4, width)],
        output_dtypes=[mx.bfloat16],
    )[
        0
    ]
    expected = module.hc_norm(x).reshape(4, width)
    mx.eval(normed, expected)
    assert mx.array_equal(normed.view(mx.uint16), expected.view(mx.uint16)).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_gated_residual_call_routes_through_fused_path(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    x = mx.random.normal((1, 2, WIDTH)).astype(mx.bfloat16)
    calls = []
    original = hc_fused.fused_forward
    monkeypatch.setattr(
        hc_fused, "fused_forward", lambda m, h: calls.append(h.shape) or original(m, h)
    )
    out = module(x)
    mx.eval(out)
    assert calls == [(1, 2, WIDTH)]


@pytest.mark.parametrize("hidden,bits", [(800, 4), (1056, 5)])
def test_compatible_rejects_hidden_not_multiple_of_64(hidden, bits):
    # 64 is the quantisation group and the up kernel's grid unit; nothing below it is handled.
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(bits, True, hidden=hidden)
    x = mx.random.normal((1, 4, HC * hidden)).astype(mx.bfloat16)
    assert not hc_fused.compatible(module, x)


def test_ineligible_model_is_logged_once(monkeypatch, caplog):
    from mlx_vlm.models.qwen4_exp import hc_fused

    monkeypatch.setattr(hc_fused, "_INELIGIBLE_LOGGED", False)
    module = _module(4, True, hidden=800)
    x = mx.random.normal((1, 4, HC * 800)).astype(mx.bfloat16)
    with caplog.at_level("INFO", logger=hc_fused.logger.name):
        assert not hc_fused.compatible(module, x)
        assert not hc_fused.compatible(module, x)
        # Prefill-sized inputs are expected to skip the fused path and must not log.
        monkeypatch.setattr(hc_fused, "_INELIGIBLE_LOGGED", False)
        assert not hc_fused.compatible(
            _module(4), mx.random.normal((1, 64, WIDTH)).astype(mx.bfloat16)
        )
    messages = [
        r.getMessage()
        for r in caplog.records
        if "fused hyper-connection kernels not used" in r.getMessage()
    ]
    assert len(messages) == 1
    assert "hidden_size=800" in messages[0]


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_fused_path_takes_precedence_over_compiled_forward(monkeypatch):
    # Fused dispatch takes precedence over the compiled single-token path.
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    module._compiled_forward = lambda h: pytest.fail(
        "compiled single-token path must not run"
    )
    x = mx.random.normal((1, 1, WIDTH)).astype(mx.bfloat16)
    assert hc_fused.compatible(module, x)
    calls = []
    original = hc_fused.fused_forward
    monkeypatch.setattr(
        hc_fused, "fused_forward", lambda m, h: calls.append(h.shape) or original(m, h)
    )
    out = module(x)
    mx.eval(out)
    assert calls == [(1, 1, WIDTH)]


def test_compatible_fails_closed():
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    ok = mx.random.normal((1, 4, WIDTH)).astype(mx.bfloat16)
    if mx.metal.is_available():
        assert hc_fused.compatible(module, ok)
    assert not hc_fused.compatible(
        module, mx.random.normal((1, 17, WIDTH)).astype(mx.bfloat16)
    )
    assert not hc_fused.compatible(
        module, mx.random.normal((2, 9, WIDTH)).astype(mx.bfloat16)
    )
    assert not hc_fused.compatible(
        module, mx.random.normal((1, 4, WIDTH)).astype(mx.float16)
    )
    assert not hc_fused.compatible(
        module, mx.random.normal((4, WIDTH)).astype(mx.bfloat16)
    )
    combined = _module(4)
    combined.input_inject_weight = nn.Linear(WIDTH, LOWRANK + HC, bias=False)
    assert not hc_fused.compatible(combined, ok)
    plain_down = _module(4)
    plain_down.input_mix_weight_down = nn.Linear(WIDTH, LOWRANK, bias=False)
    assert not hc_fused.compatible(plain_down, ok)
    # Weights are immutable once planned; replacing a tensor requires rebuild_plan.
    module = _module(4)
    if mx.metal.is_available():
        assert hc_fused.compatible(module, ok)
    module.input_mix_weight_up.scales = module.input_mix_weight_up.scales.astype(
        mx.float16
    )
    hc_fused.rebuild_plan(module)
    assert not hc_fused.compatible(module, ok)
    # An FP16 layout takes FP16 rows only.
    fp16 = _module(4, dtype=mx.float16)
    if mx.metal.is_available():
        assert hc_fused.compatible(fp16, ok.astype(mx.float16))
    assert not hc_fused.compatible(fp16, ok)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize(
    "two_launch,source_name",
    [
        (False, "_N_SOURCE"),
        (False, "_DOWN"),
        (False, "_UP"),
        (True, "_NORM_DOWN"),
        (True, "_DECODE_UP"),
    ],
)
@pytest.mark.parametrize("use_combine", [False, True])
def test_lazy_compilation_failure_raises_named(
    monkeypatch, two_launch, source_name, use_combine
):
    """A kernel that fails to compile is a defect, not an eligibility
    decision: the fused path raises naming the kernel's Metal error instead
    of silently returning the canonical output."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    monkeypatch.setattr(hc_fused, "_DECODE_V2", two_launch)
    monkeypatch.setattr(hc_fused, "_KERNELS", {})
    monkeypatch.setattr(hc_fused, "_decode_launch", hc_fused._decode_launch.__wrapped__)
    monkeypatch.setattr(hc_fused, "_VALIDATED", set())
    monkeypatch.setattr(
        hc_fused,
        source_name,
        getattr(hc_fused, source_name) + "\nintentional_compile_error;\n",
    )
    module = _module(4, use_combine, hidden=64)
    x = mx.ones((1, 1, HC * 64), dtype=mx.bfloat16)
    with pytest.raises(hc_fused.FaceKernelError, match="intentional_compile_error"):
        hc_fused.fused_forward(module, x)
    assert not hc_fused._VALIDATED
    assert hc_fused.enabled()


def test_plugin_face_compile_failure_raises_named():
    """A plugin face whose Metal does not compile raises at the first launch,
    naming the face, and never falls back to the canonical path."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _plugin_module(8, broken=True)
    x = (mx.random.normal((1, 5, WIDTH)) * 2).astype(mx.bfloat16)
    with pytest.raises(hc_fused.FaceKernelError, match="hc_toy8") as failure:
        hc_fused.fused_forward(module, x)
    assert "intentional_compile_error" in str(failure.value)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_specializations_validate_once_and_keep_warm_calls_lazy(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    monkeypatch.setattr(hc_fused, "_VALIDATED", set())
    # Cover each varying template input, including mixed projection bit widths.
    for hidden, rows, down_bits, up_bits, inject_bits in [
        (64, 1, 4, 4, 4),
        (64, 4, 4, 4, 4),
        (128, 4, 4, 4, 4),
        (128, 4, 5, 4, 4),
        (128, 4, 5, 6, 4),
        (128, 4, 5, 6, 8),
        (128, 4, 5, 6, None),
    ]:
        module = _module(down_bits, inject_bits is not None, hidden=hidden)
        module.input_mix_weight_up = _module(up_bits, hidden=hidden).input_mix_weight_up
        if inject_bits is not None:
            module.block_inject_weight = _module(
                inject_bits, hidden=hidden
            ).block_inject_weight
        hc_fused.rebuild_plan(module)
        x = mx.ones((1, rows, HC * hidden), dtype=mx.bfloat16)
        mx.eval(x)
        with monkeypatch.context() as patch:
            evaluate = Mock(wraps=mx.eval)
            patch.setattr(mx, "eval", evaluate)
            first = module(x)
            second = module(x)
            assert evaluate.call_count == 1
        mx.eval(first, second)
    assert len(hc_fused._VALIDATED) == 7


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize(
    "group_size,dtype", [(32, mx.bfloat16), (64, mx.bfloat16), (64, mx.float16)]
)
@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("rows", [17, 2048])
@pytest.mark.parametrize("use_combine", [True, False])
def test_prefill_path_matches_canonical(
    bits, rows, use_combine, group_size, dtype, monkeypatch
):
    """Prefill fuses the stream norm and, with inject weights, the mix/inject tail."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(bits, use_combine, group_size=group_size, dtype=dtype)
    x = (mx.random.normal((1, rows, WIDTH)) * 2).astype(dtype)
    mx.eval(x)
    assert not hc_fused.compatible(module, x)
    assert hc_fused.prefill_compatible(module, x)
    tail = Mock(wraps=hc_fused._tail)
    monkeypatch.setattr(hc_fused, "_tail", tail)
    out = hc_fused.prefill_forward(module, x)
    assert out is not None
    assert tail.called is not use_combine
    canon = module._forward(x)
    if use_combine:
        mixed, passthrough, inject = out
        canon_mixed, _, canon_inject = canon
        assert passthrough is x
        assert inject.shape == canon_inject.shape == (1, rows, HC)
        assert _ulps(inject, canon_inject)[0] <= 4
    else:
        mixed, canon_mixed = out, canon
    assert mixed.shape == canon_mixed.shape == (1, rows, HIDDEN)
    assert mixed.dtype == dtype
    max_ulps, mean_ulps = _ulps(mixed, canon_mixed)
    assert max_ulps <= 16 and mean_ulps <= 0.5


# The prefill tail/inject arithmetic as originally shipped: one row per threadgroup, the
# normed row read from device memory for both the mix and the inject dot. Kernel changes
# must keep these bits (prefix caches and accuracy baselines depend on them).
_REFERENCE_TAIL_INJECT = r"""
    const uint row = threadgroup_position_in_grid.z;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const device T* up_r = up + (size_t)row * K;
    const device T* xn_r = xn + (size_t)row * K;
    for (int h = int(t); h < H; h += 256) {
        float acc = 0.0f;
        for (int s = 0; s < HC; ++s) {
            const int n = s * H + h;
            const T g = T(1.0f / (1.0f + metal::exp(-float(up_r[n]))));
            const T p = T(float(g) * float(xn_r[n]));
            acc = s == 0 ? float(p) : float(T(acc + float(p)));
        }
        mixed[(size_t)row * H + h] = T(acc * (1.0f / float(HC)));
    }
    constexpr int PF = hc_pack_factor<BITS_I>();
    constexpr int BP = hc_bytes_per_pack<BITS_I>();
    constexpr int ROW_BYTES = K * BP / PF;
    constexpr int GROUPS = K / GS_I;
    constexpr int PER = K / 256;
    float res[HC] = {0.0f};
    float xv[PF];
    const int e0 = int(t) * PER;
    for (int e = e0; e < e0 + PER; e += PF) {
        const float sum = hc_load_vector<T, PF, BITS_I>(xn_r + e, xv);
        const int g = e / GS_I;
        for (int r = 0; r < HC; ++r) {
            res[r] += hc_qdot<PF, BITS_I>(
                (const device uint8_t*)inject_w + r * ROW_BYTES + e * BP / PF,
                xv, float(inject_s[r * GROUPS + g]), float(inject_b[r * GROUPS + g]),
                sum);
        }
    }
    threadgroup float part[HC][8];
    for (int r = 0; r < HC; ++r) {
        const float v = simd_sum(res[r]);
        if (lane == 0) part[r][sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < HC) {
        float v = 0.0f;
        for (int i = 0; i < 8; ++i) v += part[t][i];
        const float q = float(T(float(T(v)) / float(HC)));
        const float gate = float(T(1.0f / (1.0f + metal::exp(-q))));
        inj[(size_t)row * HC + t] = T(2.0f * gate);
    }
"""


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("rows,seed", [(17, 0), (300, 1), (300, 2)])
def test_prefill_tail_inject_keeps_reference_bits(bits, rows, seed, group_size):
    from mlx_vlm.models.qwen4_exp import hc_fused

    mx.random.seed(900 + seed * 10 + bits + group_size)
    module = _module(bits, group_size=group_size)
    inject = module.block_inject_weight
    up = (mx.random.normal((rows, WIDTH)) * 3).astype(mx.bfloat16)
    normed = (mx.random.normal((rows, WIDTH)) * 2).astype(mx.bfloat16)

    expected = hc_fused._kernel(
        "test_reference_tail_inject",
        ["up", "xn", "inject_w", "inject_s", "inject_b"],
        ["mixed", "inj"],
        _REFERENCE_TAIL_INJECT,
        header=hc_fused._HEADER,
    )(
        inputs=[up, normed, inject.weight, inject.scales, inject.biases],
        template=[
            ("T", mx.bfloat16),
            ("BITS_I", bits),
            ("GS_I", group_size),
            ("K", WIDTH),
            ("H", HIDDEN),
            ("HC", HC),
        ],
        grid=(256, 1, rows),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, HIDDEN), (rows, HC)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
    )
    face = hc_fused._projection_face(inject, WIDTH, HC, mx.bfloat16, True)
    actual = hc_fused._tail_inject_launch(HC, HIDDEN, mx.bfloat16, face)(
        rows, up, normed, (inject.weight, inject.scales, inject.biases)
    )
    mx.eval(expected, actual)
    for observed, reference in zip(actual, expected):
        assert mx.array_equal(
            observed.view(mx.uint16), reference.view(mx.uint16)
        ).item()


def test_prefill_activation_rounds_like_eager_ops():
    from mlx_vlm.models.qwen4_exp import hc_fused

    mx.random.seed(5)
    special = mx.array(
        [
            0.0,
            -0.0,
            1e-40,
            -1e-40,
            3e38,
            -3e38,
            float("inf"),
            -float("inf"),
            float("nan"),
        ]
    )
    for scale in (1e-38, 1e-3, 1.0, 40.0, 1e5):
        y = (
            mx.concatenate([mx.random.normal((4096,)) * scale, special])
            .astype(mx.bfloat16)
            .reshape(1, -1, 1)
        )
        expected = nn.silu(y / HC)
        actual = hc_fused._act(HC)(y)
        mx.eval(expected, actual)
        assert mx.array_equal(actual.view(mx.uint16), expected.view(mx.uint16)).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 2), (1, 3), (1, 4), (2, 2), (1, 16), (1, 17), (1, 2048)]
)
@pytest.mark.parametrize("use_combine", [True, False])
def test_pending_write_matches_eager_write(bits, shape, use_combine, group_size):
    """The write-norm kernels store the eager residual bit for bit and normalize the same bits."""
    from mlx_vlm.models.qwen4_exp import hc_fused, language

    mx.random.seed(11 + 31 * bits + 7 * shape[0] + shape[1] + group_size)
    module = _module(bits, use_combine, group_size=group_size)
    hyper = (mx.random.normal((*shape, WIDTH)) * 2).astype(mx.bfloat16)
    branch = mx.random.normal((*shape, HIDDEN)).astype(mx.bfloat16)
    gate = (2 * mx.sigmoid(mx.random.normal((*shape, HC)))).astype(mx.bfloat16)
    written = language._hc_write(hyper, branch, gate)
    decode = shape[0] * shape[1] <= hc_fused.MAX_ROWS
    forward = hc_fused.fused_forward if decode else hc_fused.prefill_forward
    expected = forward(module, written)
    actual = forward(module, hyper, (branch, gate))
    expected = (
        (expected[0], written, expected[2]) if use_combine else (expected, written)
    )
    assert actual is not None and len(actual) == len(expected)
    mx.eval(actual, expected)
    for observed, reference in zip(actual, expected):
        assert observed.shape == reference.shape
        assert mx.array_equal(
            observed.view(mx.uint16), reference.view(mx.uint16)
        ).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("rows", [2, 64])
@pytest.mark.parametrize("promote", ["branch", "gate"])
def test_promoting_pending_write_falls_back_to_eager_write(rows, promote):
    # An FP32 branch or gate promotes the eager write; the BF16 kernels must not take it.
    from mlx_vlm.models.qwen4_exp import language

    mx.random.seed(rows)
    module = _module(6)
    hyper = mx.random.normal((1, rows, WIDTH)).astype(mx.bfloat16)
    branch = mx.random.normal((1, rows, HIDDEN))
    gate = 2 * mx.sigmoid(mx.random.normal((1, rows, HC)))
    if promote == "branch":
        gate = gate.astype(mx.bfloat16)
    else:
        branch = branch.astype(mx.bfloat16)
    expected = module(language._hc_write(hyper, branch, gate))
    actual = module(hyper, write=(branch, gate))
    mx.eval(actual, expected)
    assert actual[1].dtype == mx.float32
    for observed, reference in zip(actual, expected):
        assert mx.array_equal(observed, reference).item()


def test_prefill_path_not_offered_for_fused_rows():
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    x = mx.zeros((1, hc_fused.MAX_ROWS, WIDTH), dtype=mx.bfloat16)
    assert not hc_fused.prefill_compatible(module, x)


def test_prefill_path_kill_switch(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    x = mx.zeros((1, 64, WIDTH), dtype=mx.bfloat16)
    assert hc_fused.prefill_compatible(module, x)
    monkeypatch.setattr(hc_fused, "_DISABLED", True)
    assert not hc_fused.prefill_compatible(module, x)


def test_module_routes_prefill_rows_through_the_prefill_path(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    x = mx.zeros((1, 64, WIDTH), dtype=mx.bfloat16)
    calls = []
    monkeypatch.setattr(
        hc_fused, "prefill_forward", lambda m, h: calls.append(h.shape) or "prefill"
    )
    assert module(x) == "prefill"
    assert calls == [(1, 64, WIDTH)]
    assert module(x, target_verify=True) != "prefill"


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("path", ["prefill", "decode"])
def test_transient_failure_raises_and_recovers(monkeypatch, path):
    """A launch failure is a defect, not eligibility: it raises, and once it
    clears the same model serves again without disabling anything."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    mx.random.seed(4242)
    failed = _module(4, hidden=64)
    other = _module(8, hidden=64)
    decode = mx.ones((1, 4, HC * 64), dtype=mx.bfloat16)
    inputs = mx.ones((1, 32 if path == "prefill" else 4, HC * 64), dtype=mx.bfloat16)
    hook = "_tail" if path == "prefill" else "_kernel_norm_down"
    with monkeypatch.context() as fault:
        fault.setattr(
            hc_fused, hook, Mock(side_effect=RuntimeError("transient failure"))
        )
        with pytest.raises(
            (RuntimeError, hc_fused.FaceKernelError), match="transient failure"
        ):
            (
                hc_fused.prefill_forward(failed, inputs)
                if path == "prefill"
                else hc_fused.fused_forward(failed, decode)
            )
        assert hc_fused.enabled()

    assert hc_fused.compatible(other, decode)
    assert hc_fused.compatible(failed, decode)
    if path == "prefill":
        assert hc_fused.prefill_compatible(failed, inputs)
    # An identically seeded module serves the same bits: the failure left no
    # process-wide state behind.
    mx.random.seed(4242)
    twin = _module(4, hidden=64)
    serve = lambda module: (  # noqa: E731 - one call, two modules
        hc_fused.prefill_forward(module, inputs)
        if path == "prefill"
        else hc_fused.fused_forward(module, decode)
    )
    actual, expected = serve(failed), serve(twin)
    mx.eval(actual, expected)
    if not isinstance(actual, tuple):
        actual = (actual,)
    if not isinstance(expected, tuple):
        expected = (expected,)
    for value, reference in zip(actual, expected):
        assert mx.array_equal(value, reference).item()
    monkeypatch.setattr(
        failed, "_forward", Mock(side_effect=AssertionError("Unexpected fallback"))
    )
    monkeypatch.setattr(
        other, "_forward", Mock(side_effect=AssertionError("Unexpected fallback"))
    )
    mx.eval(failed(inputs))
    mx.eval(other(decode, target_verify=True))


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("batch,length", [(1, 3), (2, 2), (2, 8)])
def test_fused_rows_match_independent_singletons(bits, batch, length, group_size):
    from mlx_vlm.models.qwen4_exp import hc_fused

    mx.random.seed(3770 + bits + group_size)
    module = _module(bits, group_size=group_size)
    x = mx.random.normal((batch, length, WIDTH)).astype(mx.bfloat16)
    actual, _, actual_injection = hc_fused.fused_forward(module, x)
    singletons = [
        hc_fused.fused_forward(module, row.reshape(1, 1, WIDTH))
        for row in x.reshape(-1, WIDTH)
    ]
    expected = mx.concatenate([row[0] for row in singletons], axis=1).reshape(
        batch, length, HIDDEN
    )
    expected_injection = mx.concatenate([row[2] for row in singletons], axis=1).reshape(
        batch, length, HC
    )
    mx.eval(actual, expected, actual_injection, expected_injection)
    assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(actual_injection, expected_injection).item()


needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
# The tensor-unit HC prefill's format-projection eligibility is deferred
# (DEFERRED row 59, U29: hcnax, after U26/N2): the six-dispatch reference no
# longer runs MLX's quantized matmul for a FormatLinear's down/up before B10,
# so the NAX path's bitwise validation cannot hold yet. Strict, so the moment
# U29 makes these pass they fail loudly and the marker is removed. The NAX
# kernels stay covered by tests/test_qwen4_hc_prefill_nax_oracle.py, which
# drives them directly.
nax_deferred = pytest.mark.xfail(
    strict=True,
    reason="DEFERRED row 59 (U29): tensor-unit HC prefill awaits format-projection eligibility",
)
needs_nax = pytest.mark.skipif(
    not (mx.metal.is_available() and is_nax_available()),
    reason="requires Metal tensor units (M5)",
)


# Small quantized Qwen4ExpModel for model-level prefill checks.


def _deferred_model():
    """Three decoder layers (the middle, linear one carries PLE) whose hyper-connections
    take the fused prefill kernels: hc_count 4, 64-aligned widths, 4-bit
    group-64 projections."""
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    config = _tiny_config()
    config.text_config = dataclasses.replace(
        config.text_config,
        hidden_size=512,
        hc_count=4,
        hc_lowrank=64,
        ple_embed_dim=512,
        num_hidden_layers=3,
        layer_types=["linear_attention", "linear_attention", "full_attention"],
        ple_layer_ids=[2],
    )
    mx.random.seed(5)
    model = LanguageModel(config.text_config, config)
    nn.quantize(
        model,
        group_size=64,
        bits=4,
        class_predicate=lambda path, module: isinstance(module, nn.Linear)
        and "hyper_connection" in path,
    )
    # Checkpoint-like quantization statistics.
    for _, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            module.scales = (
                mx.abs(mx.random.normal(module.scales.shape)) * 0.01 + 0.002
            ).astype(mx.bfloat16)
            module.biases = (mx.random.normal(module.biases.shape) * 0.005).astype(
                mx.bfloat16
            )
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return _format_hc_layers(model), config


def _logits(model, inputs):
    out = model(inputs)
    out = getattr(out, "logits", out)
    mx.eval(out)
    return out


# Tensor-unit (NAX) prefill: bitwise parity with the MLX matmul path.


def _nax_module(bits_down: int, bits_up: int, bits_inject: int | None, seed: int):
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedResidual, Qwen4ExpRMSNorm

    mx.random.seed(seed)
    module = Qwen4ExpGatedResidual.__new__(Qwen4ExpGatedResidual)
    nn.Module.__init__(module)
    module.hc_count, module.hidden_size, module.hc_lowrank = HC, HIDDEN, LOWRANK
    module.hc_norm = Qwen4ExpRMSNorm(WIDTH, group_size=HIDDEN, eps=1e-6)
    module.hc_norm.weight = (mx.random.normal((WIDTH,)) * 0.05).astype(mx.bfloat16)
    module.input_mix_weight_down = _affine_format_linear(
        WIDTH, LOWRANK, bits_down, 64, mx.bfloat16
    )
    module.input_mix_weight_up = _affine_format_linear(
        LOWRANK, WIDTH, bits_up, 64, mx.bfloat16
    )
    if bits_inject is not None:
        module.block_inject_weight = _affine_format_linear(
            WIDTH, HC, bits_inject, 64, mx.bfloat16
        )
    mx.eval(module.parameters())
    return module


def _inputs(rows: int, write: bool):
    hyper = (mx.random.normal((1, rows, WIDTH)) * 2).astype(mx.bfloat16)
    pending = None
    if write:
        branch = mx.random.normal((1, rows, HIDDEN)).astype(mx.bfloat16)
        gate = (2 * mx.sigmoid(mx.random.normal((1, rows, HC)))).astype(mx.bfloat16)
        pending = (branch, gate)
    mx.eval(hyper, pending)
    return hyper, pending


def _bits(a: mx.array) -> mx.array:
    return a.view(mx.uint16) if a.dtype == mx.bfloat16 else a


def _mlx_path(monkeypatch, module, hyper, write):
    from mlx_vlm.models.qwen4_exp import hc_fused

    with monkeypatch.context() as patch:
        patch.setattr(hc_fused, "_NAX_DISABLED", True)
        out = hc_fused.prefill_forward(module, hyper, write)
        mx.eval(out)
    return out


def _spies(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_prefill_nax

    spies = {}
    for name in ("norm_inject", "down_silu", "up_tail"):
        spies[name] = Mock(wraps=getattr(hc_prefill_nax, name))
        monkeypatch.setattr(hc_prefill_nax, name, spies[name])
    return spies


@needs_nax
@nax_deferred
@pytest.mark.parametrize(
    "bits", [(4, 4, 4), (5, 5, 5), (6, 6, 6), (8, 8, 8), (4, 6, 5)]
)
@pytest.mark.parametrize("rows", [65, 3265, 4133])
@pytest.mark.parametrize("write", [False, True])
def test_prefill_is_bit_identical_to_mlx_path(monkeypatch, bits, rows, write):
    """Mixed input, written stream and inject weights match the six-dispatch path
    bit for bit; the down projection runs on the NAX kernel from 3265 rows on."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _nax_module(*bits, seed=rows + 7 * bits[0])
    hyper, write_args = _inputs(rows, write)
    reference = _mlx_path(monkeypatch, module, hyper, write_args)
    spies = _spies(monkeypatch)
    out = hc_fused.prefill_forward(module, hyper, write_args)
    mx.eval(out)
    assert spies["up_tail"].call_count == 1
    assert spies["down_silu"].call_count == (1 if rows >= 3265 else 0)
    mixed, passthrough, injection = out
    assert mixed.shape == (1, rows, HIDDEN) and injection.shape == (1, rows, HC)
    if not write:
        assert passthrough is hyper
    for observed, expected in zip(out, reference):
        assert observed.dtype == expected.dtype == mx.bfloat16
        assert mx.array_equal(_bits(observed), _bits(expected)).item()


@needs_nax
@nax_deferred
def test_batched_prefill_is_bit_identical(monkeypatch):
    """[batch, seq] rows flatten like MLX's non-batched quantized matmul."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _nax_module(4, 5, 6, seed=9)
    hyper = (mx.random.normal((2, 2048, WIDTH)) * 2).astype(mx.bfloat16)
    branch = mx.random.normal((2, 2048, HIDDEN)).astype(mx.bfloat16)
    gate = (2 * mx.sigmoid(mx.random.normal((2, 2048, HC)))).astype(mx.bfloat16)
    reference = _mlx_path(monkeypatch, module, hyper, (branch, gate))
    spies = _spies(monkeypatch)
    out = hc_fused.prefill_forward(module, hyper, (branch, gate))
    mx.eval(out)
    assert spies["down_silu"].call_count == 1
    assert out[0].shape == (2, 2048, HIDDEN) and out[2].shape == (2, 2048, HC)
    for observed, expected in zip(out, reference):
        assert mx.array_equal(_bits(observed), _bits(expected)).item()


def test_plain_qmm_nax_mirrors_mlx_dispatch():
    from mlx_vlm.models.qwen4_exp.hc_prefill_nax import plain_qmm_nax

    # Up projection (N = 10240): single 64-row tiles use MLX's split K.
    assert not plain_qmm_nax(64, WIDTH)
    assert plain_qmm_nax(65, WIDTH)
    # Down projection (N = 320): release wheels split K below 801 rows and
    # M5 source builds split on the tensor-unit path below 3265 rows.
    assert not plain_qmm_nax(800, LOWRANK)
    assert not plain_qmm_nax(3264, LOWRANK)
    assert plain_qmm_nax(3265, LOWRANK)
    assert plain_qmm_nax(8191, LOWRANK)


@needs_nax
def test_few_rows_and_missing_inject_keep_mlx_path(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    spies = _spies(monkeypatch)
    module = _nax_module(4, 4, 4, seed=1)
    hyper, _ = _inputs(64, False)
    mx.eval(hc_fused.prefill_forward(module, hyper))
    no_inject = _nax_module(4, 4, None, seed=2)
    hyper, _ = _inputs(128, False)
    mx.eval(hc_fused.prefill_forward(no_inject, hyper))
    assert not spies["norm_inject"].called


@needs_nax
def test_nax_prefill_kill_switch(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    spies = _spies(monkeypatch)
    monkeypatch.setattr(hc_fused, "_NAX_DISABLED", True)
    module = _nax_module(4, 4, 4, seed=4)
    hyper, _ = _inputs(128, False)
    mx.eval(hc_fused.prefill_forward(module, hyper))
    assert not spies["norm_inject"].called


@needs_nax
def test_group_size_32_keeps_mlx_prefill_path(monkeypatch):
    """The tensor-unit tile loop steps K one 64-wide group at a time."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    spies = _spies(monkeypatch)
    mx.random.seed(6)
    module = _module(4, group_size=32)
    hyper, _ = _inputs(128, False)
    assert hc_fused.prefill_compatible(module, hyper)
    assert not hc_fused._nax_prefill_ok(module, 128, WIDTH, module.block_inject_weight)
    mx.eval(hc_fused.prefill_forward(module, hyper))
    assert not spies["norm_inject"].called


@needs_nax
@nax_deferred
def test_kernel_failure_falls_back_to_mlx_path(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused, hc_prefill_nax

    module = _nax_module(4, 4, 4, seed=5)
    hyper, write = _inputs(128, True)
    reference = _mlx_path(monkeypatch, module, hyper, write)
    monkeypatch.setattr(hc_fused, "_NAX_BROKEN", False)
    monkeypatch.setattr(
        hc_prefill_nax, "up_tail", Mock(side_effect=RuntimeError("compile"))
    )
    out = hc_fused.prefill_forward(module, hyper, write)
    mx.eval(out)
    for observed, expected in zip(out, reference):
        assert mx.array_equal(_bits(observed), _bits(expected)).item()
    # Later calls do not retry the broken kernels.
    assert hc_fused._NAX_BROKEN
    assert not hc_fused._nax_prefill_ok(module, 128, WIDTH, module.block_inject_weight)


@needs_nax
@nax_deferred
def test_first_call_probe_keeps_mlx_path_on_mismatch(monkeypatch):
    """A specialization whose first result differs from the MLX path in any
    bit (e.g. an MLX with different quantized-matmul arithmetic) is replaced
    by the MLX result and the NAX kernels are not used again."""
    from mlx_vlm.models.qwen4_exp import hc_fused, hc_prefill_nax

    module = _nax_module(4, 4, 4, seed=6)
    hyper, write = _inputs(128, True)
    reference = _mlx_path(monkeypatch, module, hyper, write)
    real = hc_prefill_nax.up_tail

    def off_by_one_ulp(*args, **kwargs):
        mixed, inj = real(*args, **kwargs)
        return (mixed.view(mx.uint16) ^ 1).view(mx.bfloat16), inj

    monkeypatch.setattr(hc_fused, "_VALIDATED", set())
    monkeypatch.setattr(hc_fused, "_NAX_BROKEN", False)
    monkeypatch.setattr(hc_prefill_nax, "up_tail", off_by_one_ulp)
    out = hc_fused.prefill_forward(module, hyper, write)
    mx.eval(out)
    for observed, expected in zip(out, reference):
        assert mx.array_equal(_bits(observed), _bits(expected)).item()
    assert hc_fused._NAX_BROKEN


@needs_nax
@nax_deferred
def test_first_call_probe_accepts_bitwise_results(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _nax_module(5, 6, 8, seed=7)
    hyper, write = _inputs(3300, True)
    monkeypatch.setattr(hc_fused, "_VALIDATED", set())
    monkeypatch.setattr(hc_fused, "_NAX_BROKEN", False)
    mx.eval(hc_fused.prefill_forward(module, hyper, write))
    assert not hc_fused._NAX_BROKEN
    assert any(sig[0] == "prefill_nax" for sig in hc_fused._VALIDATED)


@needs_nax
@nax_deferred
def test_model_prefill_logits_are_bit_identical(monkeypatch):
    """A small quantized model's prefill logits with the NAX kernels in every
    hyper-connection (and deferred writes) match the MLX path bit for bit."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    model, config = _deferred_model()
    inputs = mx.random.randint(0, config.text_config.vocab_size, (1, 520))
    with monkeypatch.context() as patch:
        patch.setattr(hc_fused, "_NAX_DISABLED", True)
        reference = _logits(model, inputs)
    spies = _spies(monkeypatch)
    observed = _logits(model, inputs)
    # 3 layers x 2 hyper-connections; the final mixer has no inject weights.
    assert spies["up_tail"].call_count == 6
    view = mx.uint32 if observed.dtype == mx.float32 else mx.uint16
    assert mx.array_equal(observed.view(view), reference.view(view)).item()


def _decode_both_ways(hc_fused, monkeypatch, module, hyper, write):
    """fused_forward outputs from the three-launch and the two-launch decode kernels.

    The two-launch kernels run at every decode row count here, including the rows
    that normally keep the three-launch kernels for speed.
    """
    monkeypatch.setattr(hc_fused, "_V2_MAX_ROWS", hc_fused.MAX_ROWS)
    if hasattr(getattr(hc_fused, "_decode_launch", None), "__wrapped__"):
        monkeypatch.setattr(
            hc_fused, "_decode_launch", hc_fused._decode_launch.__wrapped__
        )
    outputs = []
    for two_launch in (False, True):
        monkeypatch.setattr(hc_fused, "_DECODE_V2", two_launch)
        hc_fused.rebuild_plan(module)
        out = hc_fused.fused_forward(module, hyper, write)
        assert out is not None
        outputs.append(out if isinstance(out, tuple) else (out,))
    mx.eval(outputs)
    return outputs


def _assert_same_bits(outputs):
    three_launch, two_launch = outputs
    assert len(two_launch) == len(three_launch)
    for new, old in zip(two_launch, three_launch):
        assert new.shape == old.shape and new.dtype == old.dtype
        assert mx.array_equal(new.view(mx.uint16), old.view(mx.uint16)).item()


def _two_launch_fits(hc_fused, monkeypatch, module, rows):
    monkeypatch.setattr(hc_fused, "_DECODE_V2", True)
    monkeypatch.setattr(hc_fused, "_V2_MAX_ROWS", hc_fused.MAX_ROWS)
    return hc_fused._decode_v2(
        rows,
        HC,
        module.hidden_size,
        LOWRANK,
        module.hc_lowrank
        // hc_fused._projection_face(
            module.input_mix_weight_up,
            module.hc_lowrank,
            module.hc_count * module.hidden_size,
            module.hc_norm.weight.dtype,
            False,
        ).vpt,
        "block_inject_weight" in module,
    )


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("gpu_class", ["d", "s"])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("use_combine", [True, False])
def test_two_launch_decode_keeps_three_launch_bits(
    bits, use_combine, gpu_class, monkeypatch
):
    """Mixed, written residual and injection match the three-launch kernels bit for bit at
    every decode row count, with and without a pending write (checkpoint shapes), in
    each GPU class's geometry."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    monkeypatch.setattr(hc_fused, "_gpu_class", lambda: gpu_class)
    rows_list = range(1, hc_fused.MAX_ROWS + 1) if bits == 6 else (1, 2, 3, 5, 16)
    for seed in (0, 1, 2) if bits == 6 else (0,):
        mx.random.seed(4017 + 97 * seed + bits)
        module = _module(bits, use_combine)
        for rows in rows_list:
            assert _two_launch_fits(hc_fused, monkeypatch, module, rows)
            hyper = (mx.random.normal((1, rows, WIDTH)) * 2).astype(mx.bfloat16)
            branch = mx.random.normal((1, rows, HIDDEN)).astype(mx.bfloat16)
            gate = (2 * mx.sigmoid(mx.random.normal((1, rows, HC)))).astype(mx.bfloat16)
            for write in (None, (branch, gate)):
                _assert_same_bits(
                    _decode_both_ways(hc_fused, monkeypatch, module, hyper, write)
                )


# 64: every down block is a partial tail; 768/1152/1344: partial final down, inject
# and norm blocks (see test_fused_matches_canonical_path_at_other_hidden_sizes).
@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("gpu_class", ["d", "s"])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("hidden", [64, 1152, 1344])
def test_two_launch_decode_keeps_bits_with_partial_blocks(
    hidden, bits, gpu_class, monkeypatch
):
    from mlx_vlm.models.qwen4_exp import hc_fused

    monkeypatch.setattr(hc_fused, "_gpu_class", lambda: gpu_class)
    mx.random.seed(4018 + hidden + bits)
    module = _module(bits, True, hidden=hidden)
    for rows in (1, 3, 16):
        assert _two_launch_fits(hc_fused, monkeypatch, module, rows)
        hyper = (mx.random.normal((1, rows, HC * hidden)) * 2).astype(mx.bfloat16)
        branch = mx.random.normal((1, rows, hidden)).astype(mx.bfloat16)
        gate = (2 * mx.sigmoid(mx.random.normal((1, rows, HC)))).astype(mx.bfloat16)
        for write in (None, (branch, gate)):
            _assert_same_bits(
                _decode_both_ways(hc_fused, monkeypatch, module, hyper, write)
            )


# Norm/down threadgroup (g, ks) stores strip g of stream ks, so a stream needs no
# more 256-element strips than there are row groups: with two down rows per
# simdgroup (rows >= 3) that is 320 / 16 + 1 = 21, so 5376 is the widest hidden
# size and 5440 must keep the three-launch kernels.
@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_two_launch_decode_strip_boundary(monkeypatch):
    from mlx_vlm.models.qwen4_exp import hc_fused

    mx.random.seed(4019)
    widest = _module(6, True, hidden=5376)
    assert _two_launch_fits(hc_fused, monkeypatch, widest, 3)
    hyper = (mx.random.normal((1, 3, HC * 5376)) * 2).astype(mx.bfloat16)
    _assert_same_bits(_decode_both_ways(hc_fused, monkeypatch, widest, hyper, None))
    too_wide = _module(6, True, hidden=5440)
    assert not _two_launch_fits(hc_fused, monkeypatch, too_wide, 3)
    assert _two_launch_fits(hc_fused, monkeypatch, too_wide, 1)
    hyper = (mx.random.normal((1, 3, HC * 5440)) * 2).astype(mx.bfloat16)
    _assert_same_bits(_decode_both_ways(hc_fused, monkeypatch, too_wide, hyper, None))


def _probe(source: str, epilogue: str, replacement: str) -> str:
    """A kernel source whose epilogue stores the FP32 value it would have rounded."""
    assert epilogue in source
    return source.replace(epilogue, replacement)


_D_ACT_EPILOGUE = """            v = v / float(HC);
            act[(size_t)row * R + int(tg) * 8 + int(sg) * 4 + r]
                = T(v / (1.0f + metal::exp(-v)));
"""
_D_INJ_EPILOGUE = """            v = v / float(HC);
            inj[(size_t)row * HC + r] = T(2.0f / (1.0f + metal::exp(-v)));
"""
_U_EPILOGUE = """    float gate = 1.0f / (1.0f + metal::exp(-acc));
    float v = gate * float(xn[(size_t)r * K + n]);
    v += simd_shuffle_down(v, 1);
    v += simd_shuffle_down(v, 2);
    if (s == 0) mixed[(size_t)r * H + h] = T(v / float(HC));
"""
_U2_EPILOGUE = """    float gate = 1.0f / (1.0f + metal::exp(-acc));
    float v = gate * xnv;
    v += simd_shuffle_down(v, 1);
    v += simd_shuffle_down(v, 2);
    if (s == 0) mixed[(size_t)r * H + h] = T(v / float(HC));
    if constexpr (INJ != 0) {
        if (inj_thread) inj[(size_t)r * HC + int(t)] = injv;
    }
"""
_U2_ACT_COMBINE = """        v = v / float(HC);
        acts[i] = T(v / (1.0f + metal::exp(-v)));
"""
_U2_INJ_COMBINE = """            v = v / float(HC);
            injv = T(2.0f / (1.0f + metal::exp(-v)));
"""
_N_INV = "    const float inv = metal::rsqrt(tot / float(H) + eps[0]);\n"
_NDN_INV = "    const float inv = metal::rsqrt(tot / float(H) + e);\n"


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("gpu_class", ["d", "s"])
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
@pytest.mark.parametrize("rows", [1, 3, 16])
def test_two_launch_decode_keeps_three_launch_fp32_sums(
    bits, rows, group_size, gpu_class, monkeypatch
):
    """The BF16 outputs absorb most one-ulp FP32 changes (reversing the up sum order
    changed no output bit over 128 rows), so compare the FP32 values each path rounds:
    the stream sums of squares, the down and inject slice sums
    (part[0] + part[1] + part[2] + part[3]) and the up accumulators before the gate."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    monkeypatch.setattr(hc_fused, "_V2_MAX_ROWS", hc_fused.MAX_ROWS)
    monkeypatch.setattr(hc_fused, "_gpu_class", lambda: gpu_class)
    mx.random.seed(4020 + 10 * bits + rows + group_size)
    module = _module(bits, True, group_size=group_size)
    down, up = module.input_mix_weight_down, module.input_mix_weight_up
    inject = module.block_inject_weight
    flat = (mx.random.normal((rows, WIDTH)) * 2).astype(mx.bfloat16)
    norm_inputs = [flat, module.hc_norm.weight, hc_fused._eps_array(module)]

    matrices = {
        "down": hc_fused._projection_face(down, WIDTH, LOWRANK, mx.bfloat16, False),
        "inject": hc_fused._projection_face(inject, WIDTH, HC, mx.bfloat16, True),
    }
    up_face = hc_fused._projection_face(up, LOWRANK, WIDTH, mx.bfloat16, False)
    weights = [
        down.weight,
        down.scales,
        down.biases,
        inject.weight,
        inject.scales,
        inject.biases,
    ]
    rps = hc_fused._down_rps(rows)
    normed, parts, new_tots = hc_fused._down_kernel(
        "test_hc_norm_down_sums",
        ("x", "w", "eps", "down", "inject"),
        matrices,
        ("xn", "parts", "tots"),
        _probe(
            hc_fused._NORM_DOWN,
            _NDN_INV,
            _NDN_INV + "    if (t == 0 && g == 0) tots[(size_t)row * HC + s] = tot;\n",
        ),
    )(
        inputs=[*norm_inputs, *weights],
        template=[
            ("T", mx.bfloat16),
            ("K", WIDTH),
            ("H", HIDDEN),
            ("R", LOWRANK),
            ("HC", HC),
            ("INJ", 1),
            ("RPS", rps),
        ],
        grid=(256, HC * hc_fused._down_row_groups(LOWRANK, True, rps), rows),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, WIDTH), (rows, HC, LOWRANK + HC), (rows, HC)],
        output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
    )
    old_normed, old_tots = mx.fast.metal_kernel(
        name="test_hc_norm_sums",
        input_names=["x", "w", "eps"],
        output_names=["xn", "tots"],
        source=_probe(
            hc_fused._N_SOURCE,
            _N_INV,
            _N_INV + "    if (t == 0) tots[(size_t)row * 4 + s] = tot;\n",
        ),
    )(
        inputs=norm_inputs,
        template=[("T", mx.bfloat16), ("K", WIDTH), ("H", HIDDEN)],
        grid=(256, HC, rows),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, WIDTH), (rows, HC)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )
    mx.eval(normed, old_normed, new_tots, old_tots)
    assert mx.array_equal(new_tots.view(mx.uint32), old_tots.view(mx.uint32)).item()
    assert mx.array_equal(normed.view(mx.uint16), old_normed.view(mx.uint16)).item()
    down_inputs = [old_normed, *weights]
    down_names = ("xn", "down", "inject")
    down_launch = dict(
        template=[
            ("T", mx.bfloat16),
            ("K", WIDTH),
            ("R", LOWRANK),
            ("HC", HC),
            ("INJ", 1),
        ],
        grid=(32, 8 * (LOWRANK // 8 + 1), rows),
        threadgroup=(32, 8, 1),
    )
    probe = _probe(
        _probe(
            hc_fused._DOWN,
            _D_ACT_EPILOGUE,
            "            act[(size_t)row * R + int(tg) * 8 + int(sg) * 4 + r] = v;\n",
        ),
        _D_INJ_EPILOGUE,
        "            inj[(size_t)row * HC + r] = v;\n",
    )
    down_sums, inject_sums = hc_fused._down_kernel(
        "test_hc_down_sums",
        down_names,
        matrices,
        ("act", "inj"),
        probe,
    )(
        inputs=down_inputs,
        output_shapes=[(rows, LOWRANK), (rows, HC)],
        output_dtypes=[mx.float32, mx.float32],
        **down_launch,
    )
    old_act, _ = hc_fused._down_kernel(
        "test_hc_down",
        down_names,
        matrices,
        ("act", "inj"),
        hc_fused._DOWN,
    )(
        inputs=down_inputs,
        output_shapes=[(rows, LOWRANK), (rows, HC)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
        **down_launch,
    )

    up_template = [
        ("T", mx.bfloat16),
        ("K", WIDTH),
        ("R", LOWRANK),
        ("HC", HC),
        ("H", HIDDEN),
    ]
    store_acc = "    mixed[(size_t)r * K + n] = acc;\n"
    old_accs = hc_fused.kernels.face_kernel(
        "test_hc_up_sums",
        ("xn", "act", "up"),
        {"up": up_face},
        ("mixed",),
        "",
        _probe(hc_fused._UP, _U_EPILOGUE, store_acc),
    )(
        inputs=[old_normed, old_act, up.weight, up.scales, up.biases],
        template=up_template,
        grid=(256, HIDDEN // 64, rows),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, WIDTH)],
        output_dtypes=[mx.float32],
    )[
        0
    ]
    chunks = LOWRANK // up_face.vpt
    threads = chunks * (hc_fused._V2_UP_THREADS // chunks)
    outputs = hc_fused._up_outputs(rows)
    up2_probe = _probe(
        _probe(
            _probe(hc_fused._DECODE_UP, _U2_EPILOGUE, store_acc),
            _U2_ACT_COMBINE,
            "        if (threadgroup_position_in_grid.y == 0) sums[(size_t)r * PR + i] = v;\n"
            + _U2_ACT_COMBINE,
        ),
        _U2_INJ_COMBINE,
        "        sums[(size_t)r * PR + i] = v;\n" + _U2_INJ_COMBINE,
    )
    new_accs, _, new_sums = hc_fused.kernels.face_kernel(
        "test_hc_up2_sums",
        ("xn", "parts", "up"),
        {"up": up_face},
        ("mixed", "inj", "sums"),
        "",
        up2_probe,
    )(
        inputs=[normed, parts, up.weight, up.scales, up.biases],
        template=[
            *up_template,
            ("INJ", 1),
            ("NO", outputs),
            ("NC", chunks),
            ("TGS", threads),
        ],
        grid=(threads, WIDTH // outputs, rows),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, WIDTH), (rows, HC), (rows, LOWRANK + HC)],
        output_dtypes=[mx.float32, mx.bfloat16, mx.float32],
    )

    mx.eval(new_sums, down_sums, inject_sums, old_accs, new_accs)
    for new, old in (
        (new_sums[:, :LOWRANK], down_sums),
        (new_sums[:, LOWRANK:], inject_sums),
        (new_accs, old_accs),
    ):
        assert mx.array_equal(new.view(mx.uint32), old.view(mx.uint32)).item()


# Deferred residual writes: a real-shape stack must match eager writes bit for bit.
# Each layer's tail write is carried into the next hyper-connection norm (and the
# final mixer); the reference runs with deferred writes off.


def _stack(seed: int):
    """Four layers (DeltaNet, DeltaNet+PLE, sparse attention, DeltaNet) at the checkpoint's HC shapes."""
    from mlx_vlm.models import qwen4_exp
    from mlx_vlm.models.qwen4_exp import language

    text = qwen4_exp.TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=2560,
        num_hidden_layers=4,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_experts=16,
        num_experts_per_tok=10,
        shared_expert_intermediate_size=64,
        moe_intermediate_size=64,
        rms_norm_eps=1e-6,
        vocab_size=256,
        num_key_value_heads=2,
        max_position_embeddings=65536,
        hc_count=4,
        hc_lowrank=320,
        head_dim=64,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
        ],
        ple_layer_ids=[2],
        ple_embed_dim=64,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=64,
        indexer_budget=64,
        indexer_compress_ratio=4,
        eos_token_id=1,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [16, 8, 8],
            "rope_theta": 10_000,
            "partial_rotary_factor": 1.0,
        },
    )
    vision = qwen4_exp.VisionConfig(
        model_type="qwen4_exp",
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        out_hidden_size=32,
        num_heads=4,
        patch_size=14,
        in_channels=3,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    config = qwen4_exp.ModelConfig(
        text_config=text,
        vision_config=vision,
        model_type="qwen4_exp",
        image_token_id=250,
        video_token_id=251,
        vision_start_token_id=252,
        vision_end_token_id=253,
        vocab_size=256,
    )
    mx.random.seed(seed)
    model = language.LanguageModel(text, config)
    head = language.Qwen4ExpMTPModule(text)
    # Checkpoint layout: 6-bit layer HC with one 8-bit layer, 5-bit final mixer.
    for module, layout in (
        (
            model,
            [
                ("model.layers.3.", 8),
                ("model.layers.", 6),
                ("model.hyper_connection_mixer", 5),
            ],
        ),
        (head, [("", 6)]),
    ):
        module.set_dtype(mx.bfloat16)
        for name, sub in module.named_modules():
            if name.endswith("hc_norm"):
                sub.weight = (mx.random.normal(sub.weight.shape) * 0.05).astype(
                    mx.bfloat16
                )
        for prefix, bits in layout:
            nn.quantize(
                module,
                group_size=64,
                bits=bits,
                class_predicate=lambda path, m, prefix=prefix: isinstance(m, nn.Linear)
                and path.startswith(prefix)
                and "hyper_connection" in path,
            )
        mx.eval(module.parameters())
    return _format_hc_layers(model), _format_hc_layers(head)


def _run(monkeypatch, model, head, seed, deferred, script):
    from mlx_vlm.models.qwen4_exp import hc_fused, language

    monkeypatch.setattr(hc_fused, "_WRITE_DISABLED", not deferred)
    eager_writes = []
    real_write = language._hc_write
    monkeypatch.setattr(
        language,
        "_hc_write",
        lambda *args: eager_writes.append(args[0].shape) or real_write(*args),
    )
    rng = np.random.default_rng(seed)
    outputs = script(
        model,
        head,
        lambda batch, length: mx.array(
            rng.integers(2, 256, (batch, length)), dtype=mx.int32
        ),
    )
    mx.eval(outputs)
    return outputs, eager_writes


def _assert_identical(monkeypatch, seed, script):
    model, head = _stack(seed)
    # Separate contexts so the second run does not wrap the first run's spy.
    with monkeypatch.context() as patch:
        expected, eager = _run(patch, model, head, seed, False, script)
    with monkeypatch.context() as patch:
        actual, deferred = _run(patch, model, head, seed, True, script)
    # Only the PLE layer still materializes its incoming write.
    assert len(deferred) < len(eager)
    assert len(actual) == len(expected)
    for index, (observed, reference) in enumerate(zip(actual, expected)):
        assert observed.dtype == reference.dtype == mx.bfloat16, index
        assert observed.shape == reference.shape, index
        assert mx.array_equal(
            observed.view(mx.uint16), reference.view(mx.uint16)
        ).item(), f"output {index} differs"


def _decode_and_verify(model, head, ids):
    outputs = []
    cache = model.make_cache()
    outputs.append(model(ids(1, 17), cache=cache).logits)
    for rows in (1, 2, 3, 4):
        outputs.append(model(ids(1, rows), cache=cache).logits)
    for rows in (1, 2, 3, 4):
        # Lightning MTP verify: target-verify rows plus the pre-mixer residual
        # (a one-row window is the decode step itself, with no transaction).
        out = model(ids(1, rows), cache=cache, return_hidden=True)
        hidden = out.hidden_states[0]
        if out.gdn_states is not None:
            out.gdn_states.commit([rows])
        mixed, residual = head(hidden, ids(1, rows), model.model.embed_tokens)
        outputs += [out.logits, hidden, mixed, residual]
    sink = []
    outputs.append(
        model.model(ids(1, 3), cache=cache, hidden_sink=sink, capture_layer_ids=[0, 2])
    )
    outputs += sink
    batch = model.make_cache()
    outputs.append(model(ids(2, 9), cache=batch).logits)
    for rows in (1, 2):
        outputs.append(model(ids(2, rows), cache=batch).logits)
    return outputs


@needs_metal
def test_decode_and_verify_rows_match_eager_writes(monkeypatch):
    _assert_identical(monkeypatch, 0, _decode_and_verify)


@needs_metal
@pytest.mark.parametrize("chunk", [17, 257])
def test_prefill_chunks_match_eager_writes(monkeypatch, chunk):
    def prefill(model, head, ids):
        cache = model.make_cache()
        first = model(ids(1, chunk), cache=cache).logits
        second = model(ids(1, chunk), cache=cache).logits
        return [first, second, model(ids(1, 1), cache=cache).logits]

    _assert_identical(monkeypatch, 40 + chunk, prefill)


def test_hc_fused_refuses_unfilled_format_linear_placeholder():
    from mlx_vlm.models.qwen4_exp import hc_fused

    from omlx.formats import linear as formats_linear
    from omlx.formats import testing

    module = _module(4)
    projection = module.input_mix_weight_down
    tensors = {name: projection[name] for name in ("weight", "scales", "biases")}
    with testing.verified("qwen4_exp"):
        placeholder = formats_linear.FormatLinear(
            "affine", 4, 64, WIDTH, tensors, placeholders=True
        )
    placeholder.scales = placeholder.scales.astype(mx.bfloat16)
    placeholder.biases = placeholder.biases.astype(mx.bfloat16)
    module.input_mix_weight_down = placeholder
    with pytest.raises(ValueError, match="placeholder"):
        hc_fused.compatible(module, mx.zeros((1, 1, WIDTH), mx.bfloat16))
    with pytest.raises(ValueError, match="placeholder"):
        module(mx.zeros((1, 1, WIDTH), mx.bfloat16))


def test_hc_fused_declines_format_linear_subclass():
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    projection = module.input_mix_weight_down

    class Sub(type(projection)):
        pass

    projection.__class__ = Sub
    assert (
        hc_fused._projection_face(projection, WIDTH, LOWRANK, mx.bfloat16, False)
        is None
    )


def test_hc_fused_takes_format_linear_and_declines_plain_quantized_linear():
    """The loader builds FormatLinear; a bare MLX quantized layer is not that
    route, so the fused kernels must not take it (rule 29: one path)."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    x = mx.zeros((1, 1, WIDTH), mx.bfloat16)
    down = module.input_mix_weight_down
    assert type(down).__name__ == "FormatLinear"
    assert hc_fused.compatible(module, x)

    plain = nn.QuantizedLinear(WIDTH, LOWRANK, bias=False, group_size=64, bits=4)
    plain.weight = down["weight"]
    plain.scales = down["scales"]
    plain.biases = down["biases"]
    module.input_mix_weight_down = plain
    hc_fused.rebuild_plan(module)
    assert not hc_fused.compatible(module, x)


@pytest.mark.parametrize("bits", [2, 3])
def test_r14_two_and_three_bit_affine_hc_stay_canonical(bits):
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(bits)
    x = mx.zeros((1, 1, WIDTH), mx.bfloat16)
    assert not hc_fused.compatible(module, x)


def test_warm_decode_uses_prebuilt_plan_without_face_or_plan_rebuild(monkeypatch):
    """Faces and launches are plan-build state: a warm decode makes no
    affine.face lookup and never rebuilds the plan (rule 18/28)."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    x = mx.zeros((1, 1, WIDTH), mx.bfloat16)
    assert hc_fused.fused_forward(module, x) is not None
    calls = []
    face = hc_fused.affine.face

    def counted(*args, **kwargs):
        calls.append(args)
        return face(*args, **kwargs)

    monkeypatch.setattr(hc_fused.affine, "face", counted)
    builds = []
    real_build = hc_fused._build_plan

    def counted_build(module_):
        builds.append(module_)
        return real_build(module_)

    monkeypatch.setattr(hc_fused, "_build_plan", counted_build)
    out = hc_fused.fused_forward(module, x)
    assert out is not None
    assert calls == []
    assert builds == [], "a warm decode must not rebuild the plan"


def test_hc_fused_refuses_format_linear_with_wrong_dims():
    from mlx_vlm.models.qwen4_exp import hc_fused

    from omlx.formats import linear as formats_linear
    from omlx.formats import testing

    module = _module(4)
    k = WIDTH // 2
    with testing.verified("qwen4_exp"):
        tokens = mx.random.randint(0, 1 << 32, (LOWRANK, k * 4 // 32)).astype(mx.uint32)
        groups = (LOWRANK, k // 64)
        wrong = formats_linear.FormatLinear(
            "affine",
            4,
            64,
            k,
            {
                "weight": tokens,
                "scales": (mx.abs(mx.random.normal(groups)) * 0.01).astype(mx.bfloat16),
                "biases": (mx.random.normal(groups) * 0.005).astype(mx.bfloat16),
            },
        )
    module.input_mix_weight_down = wrong
    assert not hc_fused.compatible(module, mx.zeros((1, 1, WIDTH), mx.bfloat16))

    wide = _module(4)
    wide.input_mix_weight_down = _affine_format_linear(
        WIDTH, LOWRANK * 2, 4, 64, mx.bfloat16
    )
    assert not hc_fused.compatible(wide, mx.zeros((1, 1, WIDTH), mx.bfloat16))

    zero_rank = _module(4)
    zero_rank.hc_lowrank = 0
    assert not hc_fused.compatible(zero_rank, mx.zeros((1, 1, WIDTH), mx.bfloat16))


def test_hc_fused_layout_tracks_the_input_dtype():
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    x = mx.zeros((1, 1, WIDTH), mx.bfloat16)
    assert hc_fused.compatible(module, x)
    assert not hc_fused.compatible(module, x.astype(mx.float16))


def test_hc_fused_mutation_after_plan_requires_rebuild():
    """Weights are immutable after plan build: mutating a projection without
    rebuild_plan fails closed with an error naming rebuild_plan, and the
    explicit rebuild adopts the replacement bit for bit."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    from tests.oracles import hc_fused as oracle

    module = _module(4)
    x = (mx.random.normal((1, 5, WIDTH)) * 2).astype(mx.bfloat16)
    first = hc_fused.fused_forward(module, x)
    assert first is not None
    mx.eval(first)
    first_plan = module.__dict__["_omlx_hc_plan"]
    assert isinstance(first_plan, tuple)

    replacement = _module(8).input_mix_weight_down
    assert replacement.bits != module.input_mix_weight_down.bits
    module.input_mix_weight_down = replacement

    # Without rebuild_plan the plan no longer describes the layer: fail closed
    # rather than cross a 4-bit face with 8-bit weights.
    with pytest.raises(ValueError, match="rebuild_plan"):
        hc_fused.fused_forward(module, x)
    assert module.__dict__["_omlx_hc_plan"] is first_plan

    # Explicit rebuild_plan adopts the replacement layer.
    rebuilt = hc_fused.rebuild_plan(module)
    assert isinstance(rebuilt, tuple) and rebuilt is not first_plan
    assert rebuilt.down is replacement
    out = hc_fused.fused_forward(module, x)
    assert out is not None
    expected = oracle.decode(module, x, None, gpu_class="d")
    mx.eval(out, expected)
    for a, b in zip(out, expected):
        assert mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()


def test_hc_fused_refuses_bias_and_unfitting_plugin_faces():
    """A bias, a face not exact in the norm's dtype, and a face whose lanes
    do not divide the geometry are each refused by the layout (no mutant of
    these guards survives)."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    from omlx.formats import linear as formats_linear
    from omlx.formats import testing

    x = mx.zeros((1, 1, WIDTH), mx.bfloat16)

    module = _module(4)
    down = module.input_mix_weight_down
    tensors = {name: down[name] for name in ("weight", "scales", "biases")}
    with testing.verified("qwen4_exp"):
        biased = formats_linear.FormatLinear(
            "affine", 4, 64, WIDTH, tensors, bias=mx.zeros((LOWRANK,), mx.bfloat16)
        )
    module.input_mix_weight_down = biased
    assert not hc_fused.compatible(module, x)

    fp32_only = _plugin_module(8)
    fp32_only.input_mix_weight_down = _plugin_projection(
        _hc_toy_format(8, (mx.float32,)), WIDTH, LOWRANK, 81
    )
    assert not hc_fused.compatible(fp32_only, x)

    no_lanes = _plugin_module(8)
    no_lanes.input_mix_weight_down = _plugin_projection(
        _hc_toy_format(0), WIDTH, LOWRANK, 82
    )
    assert not hc_fused.compatible(no_lanes, x)

    odd = _plugin_module(8)
    odd.input_mix_weight_down = _plugin_projection(
        _hc_toy_format(3), WIDTH, LOWRANK, 83
    )
    assert not hc_fused.compatible(odd, x)

    # vpt=256 divides hidden (down) but not lowrank (up).
    up_odd = _plugin_module(8)
    up_odd.input_mix_weight_up = _plugin_projection(
        _hc_toy_format(256), LOWRANK, WIDTH, 84
    )
    assert not hc_fused.compatible(up_odd, x)

    # vpt=3 divides neither; its down face is fine, so the inject's own
    # guard is what refuses.
    inject_odd = _plugin_module(8, inject_vpt=3)
    assert not hc_fused.compatible(inject_odd, x)

    # The layout's table budget: a 2000-value float32 table is 8000 B, over
    # formats.TABLE_BYTES (4096), whichever kernel would stage it.
    tabled = _plugin_module(
        8,
        down_tables=(Table("down_table", mx.float32, (0.0,) * 625),),
        inject_tables=(Table("inject_table", mx.float32, (0.0,) * 500),),
    )
    assert not hc_fused.compatible(tabled, x)

    # A plugin layer with a bias is not the fused kernels' route either.
    biased_plugin = _plugin_module(8)
    biased_plugin.block_inject_weight = _plugin_projection(
        _hc_toy_format(8), WIDTH, HC, 85, bias=True
    )
    assert not hc_fused.compatible(biased_plugin, x)


# A plugin format's face, the toy of tests/test_formats_kernels.py with D11
# tensor names (no "weight", "scales" or "biases"): its codes are "qcodes"
# and its one side tensor "gains". The products are exact in float32, so the
# two- and three-launch decode geometries keep each other's bits.
_HC_TOY = r"""
namespace omlx_hc_toyKEY {
struct Dot {
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* qcodes;
  const device float* gains;

  template <int K>
  struct Lanes {
    static constant constexpr int VPT = 8;
    static constant constexpr int BLOCK = 32 * VPT;

    const device uint8_t* w;
    const device float* s;

    Lanes(Dot m, size_t row, uint lane)
        : w(m.qcodes + row * K + lane * VPT), s(m.gains + row) {}

    struct Row {
      const device uint8_t* w;
    };
    Row row(int r) const { return {w + r * K}; }
    void step() { w += BLOCK; }

    template <typename XP, int... I>
    static void copy(omlx_ints<I...>, XP x, thread float* xt) {
      ((xt[I] = float(x[I])), ...);
    }
    template <int... I>
    static float dot(omlx_ints<I...>, const device uint8_t* w, thread const float* xt) {
      return (... + ((float(w[I]) - 128.0f) * xt[I]));
    }

    __OMLX_READ_LOAD__ {
      copy(omlx_iota<VPT>(), x, xt);
      return 0;
    }
    __OMLX_READ_MADD__ {
      return acc + dot(omlx_iota<VPT>(), r.w, xt);
    }

    float finish(int r, float sum) const { return sum * s[r]; }

    using Decoded = Row;
    static Decoded decode(Row r) { return r; }
    __OMLX_READ_APPLY__ {
      return madd(acc, d, xt, aux);
    }
  };
};
}  // namespace omlx_hc_toyKEY
"""


def _hc_toy_format(vpt: int, dtypes=(), tables=(), broken=False):
    from omlx import formats
    from tests.test_formats_kernels import _TOY_READS

    table_tag = "_".join(f"{t.name}{len(t.values)}" for t in tables) or "0"

    class _ToyHcFormat(formats.QuantFormat):
        mode = f"hc_toy{vpt}_{len(dtypes)}_{table_tag}_{int(broken)}"
        codes_name = "qcodes"
        tensors = ("gains",)
        contract = formats.CONTRACT

        def dot(self, layer, k, n):
            suffix = f"{vpt}_{len(dtypes)}_{table_tag}_{int(broken)}"
            source = _HC_TOY.replace("VPT = 8", f"VPT = {vpt}").replace("KEY", suffix)
            if tables:
                decls = "\n".join(
                    f"  omlx_table<float, {len(t.values)}> {t.name};" for t in tables
                )
                source = source.replace(
                    "  const device float* gains;",
                    f"  const device float* gains;\n{decls}",
                )
            if broken:
                source = source.replace(
                    "float finish(int r, float sum) const { return sum * s[r]; }",
                    "float finish(int r, float sum) const { return "
                    "intentional_compile_error; }",
                )
            return formats.Dot(
                key=f"hc_toy{suffix}",
                source=source,
                type=f"omlx_hc_toy{suffix}::Dot",
                tensors=("gains",),
                vpt=vpt,
                dtypes=dtypes,
                tables=tables,
                reads=_TOY_READS,
            )

        def decode(self, tensors, bits, group_size):
            codes = tensors["qcodes"].astype(mx.float32)
            gains = tensors["gains"].astype(mx.float32)
            # The tensor holds at least 8 values (MLX binds smaller arrays as
            # constants); a matrix reads its own rows' gains.
            return (codes - 128.0) * gains[: codes.shape[-2]][:, None]

    return _ToyHcFormat()


def _plugin_projection(
    fmt, k: int, n: int, seed: int, bias: bool = False, bits: int = 8, extra=None
):
    from omlx.formats import linear as formats_linear
    from omlx.formats import testing

    mx.random.seed(seed)
    with testing.verified("qwen4_exp"):
        tensors = {
            "qcodes": mx.random.randint(0, 256, (n, k)).astype(mx.uint8),
            # MLX binds arrays of fewer than 8 elements as constants.
            "gains": mx.array(
                [2.0 ** (r % 5 - 2) for r in range(max(n, 8))], mx.float32
            ),
        }
        tensors.update(extra or {})
        return formats_linear.FormatLinear(
            fmt,
            bits,
            k,
            k,
            tensors,
            bias=mx.zeros((n,), mx.float32) if bias else None,
        )


def _plugin_module(
    vpt: int,
    inject_vpt=None,
    inject_tables=(),
    down_tables=(),
    up_tables=(),
    broken=False,
):
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedResidual, Qwen4ExpRMSNorm

    module = Qwen4ExpGatedResidual.__new__(Qwen4ExpGatedResidual)
    nn.Module.__init__(module)
    module.hc_count, module.hidden_size, module.hc_lowrank = HC, HIDDEN, LOWRANK
    module.hc_norm = Qwen4ExpRMSNorm(WIDTH, group_size=HIDDEN, eps=1e-6)
    module.hc_norm.weight = (mx.random.normal((WIDTH,)) * 0.05).astype(mx.bfloat16)
    fmt = _hc_toy_format(vpt, broken=broken)
    down_fmt = (
        _hc_toy_format(vpt, tables=down_tables, broken=broken) if down_tables else fmt
    )
    up_fmt = _hc_toy_format(vpt, tables=up_tables, broken=broken) if up_tables else fmt
    module.input_mix_weight_down = _plugin_projection(down_fmt, WIDTH, LOWRANK, 71)
    module.input_mix_weight_up = _plugin_projection(up_fmt, LOWRANK, WIDTH, 72)
    inject_fmt = (
        _hc_toy_format(inject_vpt, tables=inject_tables, broken=broken)
        if inject_vpt is not None
        else _hc_toy_format(vpt, tables=inject_tables, broken=broken)
    )
    module.block_inject_weight = _plugin_projection(inject_fmt, WIDTH, HC, 73)
    mx.eval(module.parameters())
    return module


def test_plugin_format_linear_decodes_through_its_codes_name():
    """A plugin's FormatLinear is read by its own codes name (D11) and its
    face: decode engages, and a row keeps its bits across the two- and
    three-launch geometries."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _plugin_module(8)
    assert "weight" not in module.input_mix_weight_down
    x = (mx.random.normal((1, 6, WIDTH)) * 2).astype(mx.bfloat16)
    short = hc_fused.fused_forward(module, x)
    assert short is not None and len(short) == 3
    longer = mx.concatenate(
        [x, (mx.random.normal((1, 1, WIDTH)) * 2).astype(mx.bfloat16)], axis=1
    )
    long = hc_fused.fused_forward(module, longer)
    assert long is not None and len(long) == 3
    for a, b in zip(short, long):
        head = b[:1, :6].reshape(a.shape)
        assert mx.array_equal(a.view(mx.uint16), head.view(mx.uint16)).item()


def test_plugin_vpt16_prefill_falls_back_to_the_compiled_tail(monkeypatch):
    """A vpt=16 plugin still decodes (2560 and 320 divide by 16) but its
    tail inject cannot take whole 256-thread packs at 10240, so prefill runs
    the compiled tail and the plugin's own inject, inside prefill_forward."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _plugin_module(16)
    calls = []
    tail = hc_fused._tail
    monkeypatch.setattr(
        hc_fused,
        "_tail",
        lambda hc, hidden: calls.append((hc, hidden)) or tail(hc, hidden),
    )
    hyper = (mx.random.normal((1, 17, WIDTH)) * 2).astype(mx.bfloat16)
    out = hc_fused.prefill_forward(module, hyper)
    assert out is not None and calls
    x = (mx.random.normal((1, 6, WIDTH)) * 2).astype(mx.bfloat16)
    assert hc_fused.fused_forward(module, x) is not None


def test_nax_prefill_declines_format_linear_without_poisoning():
    """The tensor-unit prefill repeats MLX's kernels; a format layer is not
    that route, so it must take the MLX prefill path and leave _NAX_BROKEN
    alone (spec U21: NAX stays MLX-only until N2)."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    # 3265 rows is where the tensor-unit down projection engages, so the
    # predicate would be True for a bare MLX layer: the projection's type is
    # what must decline the format layer.
    affine = _module(8, group_size=64)
    assert not hc_fused._nax_prefill_ok(affine, 3265, WIDTH, affine.block_inject_weight)
    plugin = _plugin_module(8)
    assert not hc_fused._nax_prefill_ok(plugin, 3265, WIDTH, plugin.block_inject_weight)
    if not mx.metal.is_available() or not is_nax_available():
        return
    hyper = (mx.random.normal((1, 3265, WIDTH)) * 2).astype(mx.bfloat16)
    out = hc_fused.prefill_forward(affine, hyper, None, nax=True)
    assert out is not None
    assert hc_fused._NAX_BROKEN is False


def test_concurrent_format_linear_decode_and_prefill():
    """Two threads, one affine and one plugin FormatLinear module, keep each
    layer's own bits (rule 14)."""
    import threading

    from mlx_vlm.models.qwen4_exp import hc_fused

    affine = _module(4)
    plugin = _plugin_module(8)
    x = (mx.random.normal((1, 5, WIDTH)) * 2).astype(mx.bfloat16)
    warm = (mx.random.normal((1, 17, WIDTH)) * 2).astype(mx.bfloat16)
    alone = {
        "affine": hc_fused.fused_forward(affine, x),
        "plugin": hc_fused.fused_forward(plugin, x),
        "affine_pf": hc_fused.prefill_forward(affine, warm),
        "plugin_pf": hc_fused.prefill_forward(plugin, warm),
    }
    assert all(value is not None for value in alone.values())
    barrier = threading.Barrier(2)
    results = {}

    def run(module, name):
        barrier.wait()
        results[name] = hc_fused.fused_forward(module, x)
        results[name + "_pf"] = hc_fused.prefill_forward(module, warm)
        for value in (results[name], results[name + "_pf"]):
            mx.eval(*value)

    errors = []

    def guarded(module, name):
        try:
            run(module, name)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            errors.append(exc)

    threads = [
        threading.Thread(target=guarded, args=(affine, "affine")),
        threading.Thread(target=guarded, args=(plugin, "plugin")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    for name, expected in alone.items():
        got = results[name]
        assert got is not None and len(got) == len(expected)
        for a, b in zip(got, expected):
            assert mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()


def test_plan_is_immutable_tuple_built_once_and_rebuilt_explicitly(monkeypatch):
    """The per-layer HC plan is an immutable tuple holding prebuilt decode and
    prefill launches, built once on first use (including across concurrent cold
    callers) and replaced only by an explicit rebuild_plan call."""
    import threading

    from mlx_vlm.models.qwen4_exp import hc_fused

    builds = []
    real_build = hc_fused._build_plan

    def counted_build(mod):
        builds.append(mod)
        return real_build(mod)

    monkeypatch.setattr(hc_fused, "_build_plan", counted_build)

    module = _module(4)
    x = (mx.random.normal((1, 4, WIDTH)) * 0.5).astype(mx.bfloat16)
    warm = (mx.random.normal((1, 17, WIDTH)) * 0.5).astype(mx.bfloat16)
    mx.eval(x, warm)

    # Concurrent cold callers build the plan exactly once under _PLAN_LOCK.
    barrier = threading.Barrier(2)
    errors = []

    def cold_call():
        try:
            barrier.wait()
            assert hc_fused.compatible(module, x)
            out = hc_fused.fused_forward(module, x)
            assert out is not None
            mx.eval(out)
        except BaseException as exc:  # noqa: BLE001 - re-raised on main thread
            errors.append(exc)

    threads = [threading.Thread(target=cold_call) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    assert len(builds) == 1, builds

    plan = module.__dict__["_omlx_hc_plan"]
    assert isinstance(plan, tuple)
    assert len(plan.decode_launches) == hc_fused.MAX_ROWS
    assert all(
        isinstance(pair, tuple) and len(pair) == 2 for pair in plan.decode_launches
    )
    assert callable(plan.tail_launch)
    for deleted in (
        "_omlx_hc_layout",
        "_hc_decode_layout",
        "_hc_tail_inject",
        "_omlx_hc_fused_eps",
    ):
        assert deleted not in module.__dict__
    for deleted_fn in (
        "_static_layout",
        "_layout_snapshot",
        "_CHECK_SNAP",
        "_check_static_layout",
    ):
        assert not hasattr(hc_fused, deleted_fn)

    # Subsequent decode and prefill calls reuse the same immutable plan.
    assert hc_fused.prefill_compatible(module, warm)
    assert hc_fused.prefill_forward(module, warm) is not None
    assert hc_fused.fused_forward(module, x) is not None
    assert module.__dict__["_omlx_hc_plan"] is plan
    assert len(builds) == 1

    # Replacing a projection requires explicit rebuild_plan.
    module.input_mix_weight_down = _module(8).input_mix_weight_down
    assert module.__dict__["_omlx_hc_plan"] is plan
    rebuilt = hc_fused.rebuild_plan(module)
    assert isinstance(rebuilt, tuple) and rebuilt is not plan
    assert module.__dict__["_omlx_hc_plan"] is rebuilt
    assert len(builds) == 2


def test_nax_prefill_ok_handles_plugin_zero_bits():
    """HIGH-2: a plugin's bits are not packed widths; eligibility declines
    before any pack-factor arithmetic."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _plugin_module(8)
    module.block_inject_weight = _plugin_projection(
        _hc_toy_format(8), WIDTH, HC, 91, bits=0
    )
    assert not hc_fused._nax_prefill_ok(module, 17, WIDTH, module.block_inject_weight)
    out = hc_fused.prefill_forward(module, mx.zeros((1, 17, WIDTH), mx.bfloat16))
    assert out is not None


def test_forward_validates_input_shape_dtype_and_rows_with_value_error():
    """fused_forward and prefill_forward validate x up front against the plan
    and raise ValueError on mismatch instead of silently computing wrong output."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    good_decode = mx.zeros((1, 4, WIDTH), mx.bfloat16)
    assert hc_fused.fused_forward(module, good_decode) is not None

    for bad_input, pattern in (
        ("not-an-array", "mx.array"),
        (mx.zeros((4, WIDTH), mx.bfloat16), "3-D"),
        (mx.zeros((1, 1, 4, WIDTH), mx.bfloat16), "3-D"),
        (mx.zeros((1, 4, WIDTH // 2), mx.bfloat16), "width"),
        (mx.zeros((1, 4, WIDTH), mx.float16), "dtype"),
        (mx.zeros((1, 4, WIDTH), mx.float32), "dtype"),
        (mx.zeros((1, 0, WIDTH), mx.bfloat16), "at least 1 row"),
        (mx.zeros((1, hc_fused.MAX_ROWS + 1, WIDTH), mx.bfloat16), "at most"),
    ):
        with pytest.raises(ValueError, match=pattern):
            hc_fused.fused_forward(module, bad_input)

    for bad_input, pattern in (
        ("not-an-array", "mx.array"),
        (mx.zeros((17, WIDTH), mx.bfloat16), "3-D"),
        (mx.zeros((1, 17, WIDTH // 2), mx.bfloat16), "width"),
        (mx.zeros((1, 17, WIDTH), mx.float16), "dtype"),
        (mx.zeros((1, 17, WIDTH), mx.float32), "dtype"),
        (mx.zeros((1, 0, WIDTH), mx.bfloat16), "at least 1 row"),
    ):
        with pytest.raises(ValueError, match=pattern):
            hc_fused.prefill_forward(module, bad_input)

    # FP32 plan is supported for decode (including pending writes) and refused
    # with ValueError on prefill_forward.
    fp32_mod = _module(4, True, dtype=mx.float32)
    with pytest.raises(ValueError, match="bfloat16 or float16"):
        hc_fused.prefill_forward(fp32_mod, mx.zeros((1, 17, WIDTH), mx.float32))


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("rows", [2, 7])
def test_fused_forward_supports_fp32_pending_write(rows):
    """FP32 hyper_input + same-dtype (branch, gate) write works on both
    two-launch and three-launch decode and matches the oracle bit for bit."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    from tests.oracles import hc_fused as oracle

    mx.random.seed(9001 + rows)
    module = _module(4, True, dtype=mx.float32)
    hyper = (mx.random.normal((1, rows, WIDTH)) * 0.5).astype(mx.float32)
    branch = (mx.random.normal((1, rows, HIDDEN)) * 0.5).astype(mx.float32)
    gate = (2 * mx.sigmoid(mx.random.normal((1, rows, HC)))).astype(mx.float32)
    # Served compatibility stays 16-bit only; direct fused_forward supports FP32.
    assert not hc_fused.compatible(module, hyper)
    got = hc_fused.fused_forward(module, hyper, (branch, gate))
    assert got is not None and len(got) == 3
    expected = oracle.decode(
        module, hyper, (branch, gate), gpu_class=hc_fused._gpu_class()
    )
    mx.eval(got, expected)
    for actual, reference in zip(got, expected):
        assert actual.dtype == mx.float32
        assert mx.array_equal(actual.view(mx.uint32), reference.view(mx.uint32)).item()


def test_nax_eligibility_refuses_every_projection_and_the_head_is_formatted():
    """MEDIUM-2: rule 29 (no bare MLX layer is eligible) and the head's HC
    layers are the loader's layers too."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    bare = nn.QuantizedLinear(WIDTH, HC, bias=False, group_size=64, bits=8)
    assert hc_fused._nax_projection_eligible(bare) is False
    model, head = _stack(7)
    gates = [
        module
        for path, module in head.named_modules()
        if "hyper_connection" in path and hasattr(module, "hc_count")
    ]
    assert gates, "the head has no hyper-connection module"
    assert hc_fused.compatible(gates[0], mx.zeros((1, 1, WIDTH), mx.bfloat16))
    assert not any(
        isinstance(module, nn.QuantizedLinear)
        for path, module in head.named_modules()
        if "hyper_connection" in path
    )


def test_face_kernel_error_and_plan_exports_and_refusal_leaves_no_plan():
    """Public exports include FaceKernelError and rebuild_plan(s); a raising
    placeholder build leaves no cached plan behind."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    from omlx.formats import linear as formats_linear
    from omlx.formats import testing

    assert "FaceKernelError" in hc_fused.__all__
    assert "rebuild_plan" in hc_fused.__all__
    assert "rebuild_plans" in hc_fused.__all__
    module = _module(4)
    down = module.input_mix_weight_down
    tensors = {name: down[name] for name in ("weight", "scales", "biases")}
    with testing.verified("qwen4_exp"):
        placeholder = formats_linear.FormatLinear(
            "affine", 4, 64, WIDTH, tensors, placeholders=True
        )
    placeholder.scales = placeholder.scales.astype(mx.bfloat16)
    placeholder.biases = placeholder.biases.astype(mx.bfloat16)
    module.input_mix_weight_down = placeholder
    with pytest.raises(ValueError, match="placeholder"):
        hc_fused.compatible(module, mx.zeros((1, 1, WIDTH), mx.bfloat16))
    assert "_omlx_hc_plan" not in module.__dict__
    assert "_omlx_hc_fused_eps" not in module.__dict__


def test_prefill_tail_inject_compile_failure_raises_named(monkeypatch):
    """LOW-2: the prefill tail inject names a broken face's error, wraps
    launch RuntimeErrors, and re-raises FaceKernelError unchanged."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _plugin_module(8, broken=True)
    with pytest.raises(hc_fused.FaceKernelError, match="hc_toy8") as failure:
        hc_fused.prefill_forward(module, mx.zeros((1, 17, WIDTH), mx.bfloat16))
    assert str(failure.value).count("failed to build or launch") == 1

    class _BoomKernel:
        def __init__(self, exc):
            self._exc = exc
            self.native = self

        def __call__(self, *args, **kwargs):
            raise self._exc

    good = _plugin_module(8)
    x = mx.zeros((1, 17, WIDTH), mx.bfloat16)
    monkeypatch.setattr(
        hc_fused.kernels,
        "face_kernel",
        lambda *args, **kwargs: _BoomKernel(RuntimeError("boom")),
    )
    with pytest.raises(
        hc_fused.FaceKernelError, match=r"hc_toy8.*prefill tail-inject.*boom"
    ):
        hc_fused.prefill_forward(good, x)

    good2 = _plugin_module(8)
    monkeypatch.setattr(
        hc_fused.kernels,
        "face_kernel",
        lambda *args, **kwargs: _BoomKernel(hc_fused.FaceKernelError("orig")),
    )
    with pytest.raises(hc_fused.FaceKernelError) as direct:
        hc_fused.prefill_forward(good2, x)
    assert str(direct.value) == "orig"


def test_decode_launch_wraps_runtime_error_and_preserves_face_kernel_error(
    monkeypatch,
):
    """Decode launch wraps non-FaceKernelError exceptions naming the faces and
    preserves an existing FaceKernelError unchanged."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    class _BoomKernel:
        def __init__(self, exc):
            self._exc = exc
            self.native = self

        def __call__(self, *args, **kwargs):
            raise self._exc

    monkeypatch.setattr(hc_fused, "_decode_launch", hc_fused._decode_launch.__wrapped__)
    module = _plugin_module(8)
    x = mx.zeros((1, 4, WIDTH), mx.bfloat16)
    monkeypatch.setattr(
        hc_fused.kernels,
        "face_kernel",
        lambda *args, **kwargs: _BoomKernel(RuntimeError("boom")),
    )
    with pytest.raises(hc_fused.FaceKernelError, match=r"decode faces .*hc_toy8.*boom"):
        hc_fused.fused_forward(module, x)

    module2 = _plugin_module(8)
    monkeypatch.setattr(
        hc_fused.kernels,
        "face_kernel",
        lambda *args, **kwargs: _BoomKernel(hc_fused.FaceKernelError("orig")),
    )
    with pytest.raises(hc_fused.FaceKernelError) as direct:
        hc_fused.fused_forward(module2, x)
    assert str(direct.value) == "orig"


def test_face_kernel_build_failure_raises_named_face_kernel_error(monkeypatch):
    """A build-time exception from kernels.face_kernel is wrapped with the face
    names instead of escaping raw for both decode and prefill tail-inject."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    real_face_kernel = hc_fused.kernels.face_kernel

    def boom_all(*args, **kwargs):
        raise RuntimeError("build boom")

    monkeypatch.setattr(hc_fused, "_decode_launch", hc_fused._decode_launch.__wrapped__)
    module = _plugin_module(8)
    monkeypatch.setattr(hc_fused.kernels, "face_kernel", boom_all)
    with pytest.raises(
        hc_fused.FaceKernelError, match=r"decode faces .*hc_toy8.*build boom"
    ):
        hc_fused.fused_forward(module, mx.zeros((1, 4, WIDTH), mx.bfloat16))

    def boom_tail(name, *args, **kwargs):
        if "tail_inject" in name:
            raise RuntimeError("build boom")
        return real_face_kernel(name, *args, **kwargs)

    warm = _plugin_module(8)
    monkeypatch.setattr(hc_fused.kernels, "face_kernel", boom_tail)
    with pytest.raises(
        hc_fused.FaceKernelError, match=r"hc_toy8.*prefill tail-inject.*build boom"
    ):
        hc_fused.prefill_forward(warm, mx.zeros((1, 17, WIDTH), mx.bfloat16))


def test_rebuild_plan_failure_retires_stale_plan():
    """If rebuild_plan raises (e.g. an unfilled placeholder), any previous plan
    is cleared before the raise so no stale plan survives."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    from omlx.formats import linear as formats_linear
    from omlx.formats import testing

    module = _module(4)
    x = mx.zeros((1, 4, WIDTH), mx.bfloat16)
    assert hc_fused.fused_forward(module, x) is not None
    assert isinstance(module.__dict__.get("_omlx_hc_plan"), tuple)

    down = module.input_mix_weight_down
    tensors = {name: down[name] for name in ("weight", "scales", "biases")}
    with testing.verified("qwen4_exp"):
        placeholder = formats_linear.FormatLinear(
            "affine", 4, 64, WIDTH, tensors, placeholders=True
        )
    placeholder.scales = placeholder.scales.astype(mx.bfloat16)
    placeholder.biases = placeholder.biases.astype(mx.bfloat16)
    module.input_mix_weight_down = placeholder

    with pytest.raises(ValueError, match="placeholder"):
        hc_fused.rebuild_plan(module)
    assert "_omlx_hc_plan" not in module.__dict__


def test_decode_both_ways_runs_both_launch_forms(monkeypatch):
    """_decode_both_ways must execute both 3-launch and 2-launch forms on the
    same module after rebuilding its plan."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    seen = []
    real_v2 = hc_fused._decode_v2

    def record_v2(*args, **kwargs):
        result = real_v2(*args, **kwargs)
        seen.append(result)
        return result

    monkeypatch.setattr(hc_fused, "_decode_v2", record_v2)
    _decode_both_ways(hc_fused, monkeypatch, module, x, None)
    assert set(seen) == {False, True}, seen


def test_bad_layouts_produce_false_plan_for_all_paths():
    """Every geometry, norm, face, vpt and table-budget rule is enforced once
    in _build_plan so compatible, fused_forward and prefill_forward all agree."""
    from mlx_vlm.models.qwen4_exp import hc_fused
    from mlx_vlm.models.qwen4_exp.language import (
        Qwen4ExpGatedResidual,
        Qwen4ExpRMSNorm,
    )

    def _custom_plugin(vpt, hc=HC, hidden=HIDDEN, lowrank=LOWRANK):
        width = hc * hidden
        mod = Qwen4ExpGatedResidual.__new__(Qwen4ExpGatedResidual)
        nn.Module.__init__(mod)
        mod.hc_count, mod.hidden_size, mod.hc_lowrank = hc, hidden, lowrank
        mod.hc_norm = Qwen4ExpRMSNorm(width, group_size=hidden, eps=1e-6)
        mod.hc_norm.weight = mx.zeros((width,), mx.bfloat16)
        fmt = _hc_toy_format(vpt)
        mod.input_mix_weight_down = _plugin_projection(fmt, width, lowrank, 101)
        mod.input_mix_weight_up = _plugin_projection(fmt, lowrank, width, 102)
        mod.block_inject_weight = _plugin_projection(fmt, width, hc, 103)
        mx.eval(mod.parameters())
        return mod

    cases = []

    fp32_only = _plugin_module(8)
    fp32_only.input_mix_weight_down = _plugin_projection(
        _hc_toy_format(8, (mx.float32,)), WIDTH, LOWRANK, 94
    )
    cases.append((fp32_only, WIDTH))

    odd = _plugin_module(8)
    odd.input_mix_weight_down = _plugin_projection(
        _hc_toy_format(3), WIDTH, LOWRANK, 95
    )
    cases.append((odd, WIDTH))

    up_odd = _plugin_module(8)
    up_odd.input_mix_weight_up = _plugin_projection(
        _hc_toy_format(256), LOWRANK, WIDTH, 96
    )
    cases.append((up_odd, WIDTH))

    inject_odd = _plugin_module(8, inject_vpt=3)
    cases.append((inject_odd, WIDTH))

    bare_up = _plugin_module(8)
    bare_up.input_mix_weight_up = nn.QuantizedLinear(
        LOWRANK, WIDTH, bias=False, group_size=64, bits=8
    )
    cases.append((bare_up, WIDTH))

    tabled = _plugin_module(
        8,
        down_tables=(Table("d", mx.float32, (0.0,) * 625),),
        inject_tables=(Table("i", mx.float32, (0.0,) * 500),),
    )
    cases.append((tabled, WIDTH))

    # Up table alone pushes combined table bytes over formats.TABLE_BYTES.
    up_tabled = _plugin_module(
        8,
        down_tables=(Table("d", mx.float32, (0.0,) * 512),),
        up_tables=(Table("u", mx.float32, (0.0,) * 513),),
    )
    cases.append((up_tabled, WIDTH))

    # Geometry / norm structural mismatches (must return None, not crash/corrupt).
    cases.append((_custom_plugin(8, hidden=32), HC * 32))
    cases.append((_custom_plugin(8, hidden=800), HC * 800))
    cases.append((_custom_plugin(8, hc=2), 2 * HIDDEN))
    cases.append((_custom_plugin(4, lowrank=68), WIDTH))

    zero_hidden = _plugin_module(8)
    zero_hidden.hidden_size = 0
    zero_hidden.hc_norm.group_size = 0
    zero_hidden.hc_norm.weight = mx.zeros((0,), mx.bfloat16)
    cases.append((zero_hidden, 0))

    neg_hidden = _plugin_module(8)
    neg_hidden.hidden_size = -64
    neg_hidden.hc_norm.group_size = -64
    cases.append((neg_hidden, WIDTH))

    neg_lowrank = _plugin_module(8)
    neg_lowrank.hc_lowrank = -64
    cases.append((neg_lowrank, WIDTH))

    float_hc = _plugin_module(8)
    float_hc.hc_count = 4.0
    cases.append((float_hc, WIDTH))

    float_hidden = _plugin_module(8)
    float_hidden.hidden_size = float(HIDDEN)
    cases.append((float_hidden, WIDTH))

    float_lowrank = _plugin_module(8)
    float_lowrank.hc_lowrank = float(LOWRANK)
    cases.append((float_lowrank, WIDTH))

    short_norm = _plugin_module(8)
    short_norm.hc_norm.weight = mx.zeros((HIDDEN,), mx.bfloat16)
    cases.append((short_norm, WIDTH))

    none_norm = _plugin_module(8)
    none_norm.hc_norm.weight = None
    cases.append((none_norm, WIDTH))

    bad_group = _plugin_module(8)
    bad_group.hc_norm.group_size = HIDDEN // 2
    cases.append((bad_group, WIDTH))

    combined = _plugin_module(8)
    combined.input_inject_weight = combined.input_mix_weight_down
    cases.append((combined, WIDTH))

    for module, width in cases:
        assert not hc_fused.compatible(module, mx.zeros((1, 1, width), mx.bfloat16))
        assert (
            hc_fused.fused_forward(module, mx.zeros((1, 7, width), mx.bfloat16)) is None
        )
        assert (
            hc_fused.prefill_forward(module, mx.zeros((1, 17, width), mx.bfloat16))
            is None
        )
        assert module.__dict__["_omlx_hc_plan"] is False

    # Exact table budget boundary (4096 B == formats.TABLE_BYTES) is accepted.
    exact_budget = _plugin_module(
        8,
        down_tables=(Table("d", mx.float32, (0.0,) * 512),),
        inject_tables=(Table("i", mx.float32, (0.0,) * 512),),
    )
    assert hc_fused.compatible(exact_budget, mx.zeros((1, 1, WIDTH), mx.bfloat16))
    assert (
        hc_fused.fused_forward(exact_budget, mx.zeros((1, 4, WIDTH), mx.bfloat16))
        is not None
    )


def test_plan_built_once_across_decode_and_prefill(monkeypatch):
    """A module resolves its faces and validates its layout once, then reuses
    that single immutable plan across decode and prefill."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    layout_calls = []
    face_calls = []
    real_ok = hc_fused._layout_ok
    real_face = hc_fused._projection_face

    def count_ok(*args, **kwargs):
        layout_calls.append(1)
        return real_ok(*args, **kwargs)

    def count_face(*args, **kwargs):
        face_calls.append(1)
        return real_face(*args, **kwargs)

    monkeypatch.setattr(hc_fused, "_layout_ok", count_ok)
    monkeypatch.setattr(hc_fused, "_projection_face", count_face)

    module = _module(4)
    x = mx.zeros((1, 4, WIDTH), mx.bfloat16)
    warm = mx.zeros((1, 17, WIDTH), mx.bfloat16)
    assert hc_fused.compatible(module, x)
    assert hc_fused.fused_forward(module, x) is not None
    assert hc_fused.prefill_compatible(module, warm)
    assert hc_fused.prefill_forward(module, warm) is not None
    assert len(layout_calls) == 1, layout_calls
    assert len(face_calls) == 3, face_calls


def test_plugin_extra_side_tensor_mutation_fails_closed():
    """Round 9 MEDIUM-1: a plugin tensor outside face.tensors is recorded too;
    swapping it post-load fails closed."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    base = _hc_toy_format(8)

    class _Extra(type(base)):
        mode = base.mode + "_extra"
        tensors = (*base.tensors, "levels")

    extra = _Extra()
    module = _module(4)
    down = _plugin_projection(
        extra,
        WIDTH,
        LOWRANK,
        301,
        extra={"levels": mx.ones((LOWRANK,), mx.float32)},
    )
    module.input_mix_weight_down = down
    module.__dict__.pop("_omlx_hc_plan", None)
    assert hc_fused.rebuild_plan(module) is not False
    down["levels"] = mx.zeros_like(down["levels"])
    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    warm = mx.zeros((1, 17, WIDTH), mx.bfloat16)
    with pytest.raises(ValueError, match="rebuild_plan"):
        hc_fused.fused_forward(module, x)
    with pytest.raises(ValueError, match="rebuild_plan"):
        hc_fused.prefill_forward(module, warm)


def test_projection_class_mutation_fails_closed():
    """Round 9 MEDIUM-2: an in-place __class__ rebinding of a projection is a
    recorded read and fails closed."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    plan = hc_fused.rebuild_plan(module)
    assert plan is not False
    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    warm = mx.zeros((1, 17, WIDTH), mx.bfloat16)
    down = module.input_mix_weight_down
    original = type(down)

    class Sub(original):
        pass

    down.__class__ = Sub
    try:
        with pytest.raises(ValueError, match="rebuild_plan"):
            hc_fused.fused_forward(module, x)
        with pytest.raises(ValueError, match="rebuild_plan"):
            hc_fused.prefill_forward(module, warm)
    finally:
        down.__class__ = original


def test_equal_scalar_rebinding_is_accepted():
    """Round 9 mutant: a distinct object with an equal scalar value is not
    stale (scalars compare by type and ==)."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    assert hc_fused.fused_forward(module, x) is not None
    module.hc_lowrank = int(str(module.hc_lowrank))
    assert hc_fused.fused_forward(module, x) is not None
    module.hc_norm.eps = float(f"{module.hc_norm.eps!r}")
    assert hc_fused.fused_forward(module, x) is not None
    down = module.input_mix_weight_down
    down.mode = "".join([down.mode])
    assert hc_fused.fused_forward(module, x) is not None


def test_a_deleted_recorded_tensor_fails_closed():
    """Round 9 mutant: deleting a recorded tensor key is stale, not a crash."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    assert hc_fused.fused_forward(module, x) is not None
    del module.input_mix_weight_down["scales"]
    with pytest.raises(ValueError, match="rebuild_plan"):
        hc_fused.fused_forward(module, x)


def test_projection_training_mutation_fails_closed():
    """Round 12 (a): only a training-sensitive face records the flag. An
    ordinary affine plan survives model.eval()/train(); a plugin format that
    declares training sensitivity still fails closed on a flip."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    affine = _module(4)
    affine.eval()
    assert hc_fused.rebuild_plan(affine) is not False
    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    affine.train(True)
    assert hc_fused.fused_forward(affine, x) is not None

    base = _hc_toy_format(8)

    class _Sensitive(type(base)):
        mode = base.mode + "_sensitive"
        training_sensitive = True

    fmt = _Sensitive()
    module = _module(4)
    module.input_mix_weight_down = _plugin_projection(fmt, WIDTH, LOWRANK, 501)
    module.input_mix_weight_up = _plugin_projection(fmt, LOWRANK, WIDTH, 502)
    module.block_inject_weight = _plugin_projection(fmt, WIDTH, HC, 503)
    module.eval()
    module.__dict__.pop("_omlx_hc_plan", None)
    assert hc_fused.rebuild_plan(module) is not False
    warm = mx.zeros((1, 17, WIDTH), mx.bfloat16)
    module.train(True)
    try:
        with pytest.raises(ValueError, match="rebuild_plan"):
            hc_fused.fused_forward(module, x)
        with pytest.raises(ValueError, match="rebuild_plan"):
            hc_fused.prefill_forward(module, warm)
    finally:
        module.eval()


def test_direct_load_weights_then_eval_keeps_affine_plans_usable():
    """A plan built before the caller's model.eval() stays valid across it
    for an ordinary affine format (the plan records no training state)."""
    from mlx.utils import tree_flatten
    from mlx_vlm.models.qwen4_exp import hc_fused

    model, _ = _stack(23)
    model.load_weights(list(tree_flatten(model.parameters())), strict=True)
    hc = [
        sub
        for _, sub in model.named_modules()
        if hasattr(sub, "hc_norm") and hasattr(sub, "hc_lowrank")
    ][0]
    assert hc_fused.rebuild_plan(hc) is not False
    model.eval()
    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    assert hc_fused.fused_forward(hc, x) is not None


def test_a_stale_plan_is_refused_before_input_validation():
    """Round 10/11 mutants: the plan check runs before input validation and
    before the prefill dtype gate on both entry points."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    down = module.input_mix_weight_down
    assert hc_fused.fused_forward(module, mx.zeros((1, 2, WIDTH), mx.bfloat16))
    down["scales"] = mx.zeros_like(down["scales"])
    with pytest.raises(ValueError, match="rebuild_plan"):
        hc_fused.fused_forward(module, mx.zeros((1, 2, WIDTH), mx.float32))
    with pytest.raises(ValueError, match="rebuild_plan"):
        hc_fused.prefill_forward(module, mx.zeros((1, 17, WIDTH), mx.float32))

    fp32_module = _module(4, dtype=mx.float32)
    assert hc_fused.rebuild_plan(fp32_module) is not False
    fp32_down = fp32_module.input_mix_weight_down
    fp32_down["scales"] = mx.zeros_like(fp32_down["scales"])
    with pytest.raises(ValueError, match="rebuild_plan"):
        hc_fused.prefill_forward(fp32_module, mx.zeros((1, 17, WIDTH), mx.float32))


def test_projection_face_refuses_a_biased_layer():
    """Round 10/11 LOW-1: the direct face helper refuses a biased projection
    for both built-in affine and plugin formats."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    down = module.input_mix_weight_down
    down["bias"] = mx.zeros((LOWRANK,), mx.bfloat16)
    assert hc_fused._projection_face(down, WIDTH, LOWRANK, mx.bfloat16, False) is None

    plugin_down = _plugin_module(8).input_mix_weight_down
    plugin_down["bias"] = mx.zeros((LOWRANK,), mx.bfloat16)
    assert (
        hc_fused._projection_face(plugin_down, WIDTH, LOWRANK, mx.bfloat16, False)
        is None
    )


def test_hc_norm_class_declines_subclass_and_mutation_fails_closed():
    """Round 11 MEDIUM-1: hc_norm must be an exact Qwen4ExpRMSNorm at plan
    build, and rebinding hc_norm.__class__ post-load fails closed."""
    from mlx_vlm.models.qwen4_exp import hc_fused
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpRMSNorm

    class SubNorm(Qwen4ExpRMSNorm):
        pass

    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    warm = mx.zeros((1, 17, WIDTH), mx.bfloat16)

    sub_mod = _module(4)
    sub_mod.hc_norm.__class__ = SubNorm
    assert not hc_fused.compatible(sub_mod, x)
    assert hc_fused.fused_forward(sub_mod, x) is None
    assert hc_fused.prefill_forward(sub_mod, warm) is None
    assert sub_mod.__dict__["_omlx_hc_plan"] is False

    foreign_mod = _module(4)
    foreign_norm = nn.RMSNorm(WIDTH, eps=1e-6)
    foreign_norm.weight = mx.zeros((WIDTH,), mx.bfloat16)
    foreign_norm.group_size = HIDDEN
    foreign_mod.hc_norm = foreign_norm
    assert not hc_fused.compatible(foreign_mod, x)
    assert hc_fused.fused_forward(foreign_mod, x) is None
    assert hc_fused.prefill_forward(foreign_mod, warm) is None
    assert foreign_mod.__dict__["_omlx_hc_plan"] is False

    none_mod = _module(4)
    none_mod.hc_norm = None
    assert not hc_fused.compatible(none_mod, x)
    assert hc_fused.fused_forward(none_mod, x) is None
    assert hc_fused.prefill_forward(none_mod, warm) is None
    assert none_mod.__dict__["_omlx_hc_plan"] is False

    module = _module(4)
    assert hc_fused.rebuild_plan(module) is not False
    module.hc_norm.__class__ = SubNorm
    try:
        with pytest.raises(ValueError, match="rebuild_plan"):
            hc_fused.fused_forward(module, x)
        with pytest.raises(ValueError, match="rebuild_plan"):
            hc_fused.prefill_forward(module, warm)
    finally:
        module.hc_norm.__class__ = Qwen4ExpRMSNorm


def test_module_class_declines_subclass_and_mutation_fails_closed():
    """The HC module itself must be an exact Qwen4ExpGatedResidual at plan
    build, and rebinding module.__class__ post-load fails closed."""
    from mlx_vlm.models.qwen4_exp import hc_fused
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedResidual

    class SubResidual(Qwen4ExpGatedResidual):
        pass

    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    warm = mx.zeros((1, 17, WIDTH), mx.bfloat16)

    sub_mod = _module(4)
    sub_mod.__class__ = SubResidual
    assert not hc_fused.compatible(sub_mod, x)
    assert hc_fused.fused_forward(sub_mod, x) is None
    assert hc_fused.prefill_forward(sub_mod, warm) is None
    assert sub_mod.__dict__["_omlx_hc_plan"] is False

    module = _module(4)
    assert hc_fused.rebuild_plan(module) is not False
    module.__class__ = SubResidual
    try:
        with pytest.raises(ValueError, match="rebuild_plan"):
            hc_fused.fused_forward(module, x)
        with pytest.raises(ValueError, match="rebuild_plan"):
            hc_fused.prefill_forward(module, warm)
    finally:
        module.__class__ = Qwen4ExpGatedResidual


def test_distinct_plugin_operand_dtypes_validate_separately():
    """Round 11 MEDIUM-1: two layers with the same face but different operand
    tensor dtypes must not share a specialization; the bad one raises."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    good = _plugin_module(8)
    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    warm = mx.zeros((1, 17, WIDTH), mx.bfloat16)
    assert hc_fused.fused_forward(good, x) is not None
    assert hc_fused.prefill_forward(good, warm) is not None
    bad = _plugin_module(8)
    bad.input_mix_weight_down = _plugin_projection(
        _hc_toy_format(8),
        WIDTH,
        LOWRANK,
        401,
        extra={"gains": mx.ones((max(LOWRANK, 8),), mx.float16)},
    )
    bad.block_inject_weight = _plugin_projection(
        _hc_toy_format(8),
        WIDTH,
        HC,
        402,
        extra={"gains": mx.ones((max(HC, 8),), mx.float16)},
    )
    bad.__dict__.pop("_omlx_hc_plan", None)
    with pytest.raises(hc_fused.FaceKernelError, match="hc_toy8"):
        hc_fused.fused_forward(bad, x)
    with pytest.raises(hc_fused.FaceKernelError, match="hc_toy8"):
        hc_fused.prefill_forward(bad, warm)


def test_missing_extra_plugin_side_tensor_declines_cleanly():
    """Round 11 LOW-1: a declared side tensor deleted before the build makes
    the layer ineligible instead of raising KeyError."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    base = _hc_toy_format(8)

    class _Extra(type(base)):
        mode = base.mode + "_missing_extra"
        tensors = (*base.tensors, "levels")

    module = _module(4)
    module.input_mix_weight_down = _plugin_projection(
        _Extra(),
        WIDTH,
        LOWRANK,
        403,
        extra={"levels": mx.ones((LOWRANK,), mx.float32)},
    )
    del module.input_mix_weight_down["levels"]
    module.__dict__.pop("_omlx_hc_plan", None)
    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    assert not hc_fused.compatible(module, x)
    assert hc_fused.fused_forward(module, x) is None
    assert module.__dict__["_omlx_hc_plan"] is False


def test_affine_float_bits_and_group_size_decline_cleanly():
    """Round 11/12 layout checks: float bits/group_size, float norm
    group_size, non-uint32 affine weight, and mismatched affine biases dtype
    decline cleanly instead of emitting invalid Metal."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    for name, value in (("bits", 4.0), ("group_size", 64.0)):
        module = _module(4)
        setattr(module.input_mix_weight_down, name, value)
        module.__dict__.pop("_omlx_hc_plan", None)
        assert not hc_fused.compatible(module, x)
        assert hc_fused.fused_forward(module, x) is None
        assert module.__dict__["_omlx_hc_plan"] is False

    float_norm = _module(4)
    float_norm.hc_norm.group_size = float(HIDDEN)
    float_norm.__dict__.pop("_omlx_hc_plan", None)
    assert not hc_fused.compatible(float_norm, x)
    assert hc_fused.fused_forward(float_norm, x) is None
    assert float_norm.__dict__["_omlx_hc_plan"] is False

    bad_weight = _module(4)
    dict.__setitem__(
        bad_weight.input_mix_weight_down,
        "weight",
        bad_weight.input_mix_weight_down["weight"].astype(mx.int32),
    )
    assert bad_weight.input_mix_weight_down["weight"].dtype == mx.int32
    bad_weight.__dict__.pop("_omlx_hc_plan", None)
    assert not hc_fused.compatible(bad_weight, x)
    assert hc_fused.fused_forward(bad_weight, x) is None
    assert bad_weight.__dict__["_omlx_hc_plan"] is False

    bad_biases = _module(4)
    bad_biases.input_mix_weight_down["biases"] = bad_biases.input_mix_weight_down[
        "biases"
    ].astype(mx.float16)
    assert bad_biases.input_mix_weight_down["biases"].dtype == mx.float16
    bad_biases.__dict__.pop("_omlx_hc_plan", None)
    assert not hc_fused.compatible(bad_biases, x)
    assert hc_fused.fused_forward(bad_biases, x) is None
    assert bad_biases.__dict__["_omlx_hc_plan"] is False


def test_format_missing_on_projection_declines_cleanly():
    """Round 13 mutant: a FormatLinear whose fmt is missing/non-QuantFormat
    declines cleanly without a dead mode-lookup fallback."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    object.__setattr__(module.input_mix_weight_down, "fmt", None)
    module.__dict__.pop("_omlx_hc_plan", None)
    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    assert not hc_fused.compatible(module, x)
    assert hc_fused.fused_forward(module, x) is None
    assert module.__dict__["_omlx_hc_plan"] is False


def test_non_array_extra_plugin_side_tensor_builds_plan():
    """Round 12 mutant: an extra plugin side slot holding a non-array does
    not crash operand_dtypes derivation at plan build."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    base = _hc_toy_format(8)

    class _ExtraMeta(type(base)):
        mode = base.mode + "_extra_meta"
        tensors = (*base.tensors, "levels")

    module = _module(4)
    module.input_mix_weight_down = _plugin_projection(
        _ExtraMeta(),
        WIDTH,
        LOWRANK,
        404,
        extra={"levels": mx.ones((LOWRANK,), mx.float32)},
    )
    dict.__setitem__(module.input_mix_weight_down, "levels", None)
    assert module.input_mix_weight_down["levels"] is None
    module.__dict__.pop("_omlx_hc_plan", None)
    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    assert hc_fused.rebuild_plan(module) is not False
    assert hc_fused.fused_forward(module, x) is not None


def test_compiled_plan_checks_skip_redundant_members_and_precompute_prefill():
    """Round 12 OPT-1/OPT-2 & Round 13 mutant #2: present tensor keys checked
    in id_items are omitted from plan.members, prefill_signatures are
    precomputed on the plan, and dead plan.codes is absent."""
    from pathlib import Path

    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    plan = hc_fused.rebuild_plan(module)
    assert plan is not False
    assert not hasattr(plan, "codes")
    page = (Path(__file__).parents[1] / "docs" / "quant-formats.md").read_text()
    assert "faces, codes," not in page
    assert "faces, codes," not in hc_fused.__doc__
    present_members = {(id(o), n) for o, n, v in plan.members if v}
    item_keys = {(id(o), n) for o, n, _ in plan.id_items}
    assert item_keys
    assert not (present_members & item_keys)
    # Deleting a recorded tensor still fails closed via id_items KeyError.
    saved = module.input_mix_weight_down.pop("scales")
    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    try:
        with pytest.raises(ValueError, match="rebuild_plan"):
            hc_fused.fused_forward(module, x)
    finally:
        module.input_mix_weight_down["scales"] = saved
    assert len(plan.prefill_signatures) == 2
    warm = mx.zeros((1, 17, WIDTH), mx.bfloat16)
    assert hc_fused.prefill_forward(module, warm, nax=False) is not None
    assert plan.prefill_signatures[False] in hc_fused._VALIDATED
    branch = mx.zeros((1, 17, HIDDEN), mx.bfloat16)
    gate = mx.ones((1, 17, HC), mx.bfloat16)
    assert (
        hc_fused.prefill_forward(module, warm, write=(branch, gate), nax=False)
        is not None
    )
    assert plan.prefill_signatures[True] in hc_fused._VALIDATED


def test_hc_plan_docs_match_post_eval_contract():
    """Round 14 LOW: no doc or comment still says load_weights builds plans."""
    from pathlib import Path

    from mlx_vlm.models.qwen4_exp import hc_fused

    page = (Path(__file__).parents[1] / "docs" / "quant-formats.md").read_text()
    loader = (
        Path(__file__).parents[1] / "omlx" / "patches" / "format_load.py"
    ).read_text()
    assert "from ``qwen4_exp``" not in hc_fused.__doc__
    assert "post-eval hook" in hc_fused.__doc__
    assert "built by load_weights" not in loader
    assert "loads weights" not in page
    assert "builds no HC plans, so build them once here" in loader
    assert "load patch's post-eval hook" in page
    assert "after the loader's ``model.eval()``" in hc_fused.rebuild_plans.__doc__


def test_decode_metal_kernels_use_constexpr_inj():
    """Round 14 LOW: every decode kernel's INJ branch is if constexpr."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    assert hc_fused._DOWN.count("if constexpr (INJ != 0)") >= 1
    assert "INJ != 0 &&" not in hc_fused._DECODE_UP
    assert hc_fused._DECODE_UP.count("if constexpr (INJ != 0)") >= 2
    assert "INJ != 0 &&" not in hc_fused._NORM_DOWN_TEMPLATE
    assert hc_fused._NORM_DOWN_TEMPLATE.count("else if constexpr (INJ != 0)") == 1


def test_three_launch_down_grid_omits_inject_tg_when_absent(monkeypatch):
    """Round 15 LOW-3: the 3-launch down grid drops the inject threadgroups
    when the layer has no inject projection."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    grids = []
    real_down_kernel = hc_fused._down_kernel

    class _Proxy:
        def __init__(self, kernel):
            self._kernel = kernel
            self.native = kernel.native

        def __call__(self, **kwargs):
            grids.append(kwargs.get("grid"))
            return self._kernel(**kwargs)

    def down_kernel(name, *args, **kwargs):
        kernel = real_down_kernel(name, *args, **kwargs)
        if name == "omlx_qwen4_hc_fused_down":
            return _Proxy(kernel)
        return kernel

    monkeypatch.setattr(hc_fused, "_decode_launch", hc_fused._decode_launch.__wrapped__)
    monkeypatch.setattr(hc_fused, "_down_kernel", down_kernel)
    x = mx.zeros((1, 7, WIDTH), mx.bfloat16)
    for use_combine in (False, True):
        module = _module(4, use_combine=use_combine)
        module.__dict__.pop("_omlx_hc_plan", None)
        plan = hc_fused.rebuild_plan(module)
        assert plan is not False
        assert hc_fused.fused_forward(module, x) is not None
    assert grids == [
        (32, 8 * (LOWRANK // 8), 7),
        (32, 8 * (LOWRANK // 8 + 1), 7),
    ]


def test_quant_formats_doc_documents_training_sensitive():
    """Round 13 LOW-1: docs/quant-formats.md names QuantFormat.training_sensitive
    both in the Faces section and in the hyper-connection paragraph."""
    from pathlib import Path

    page = (Path(__file__).parents[1] / "docs" / "quant-formats.md").read_text()
    faces_start = page.index("## Faces")
    faces_end = page.index("\n## ", faces_start + 1)
    assert "training_sensitive" in page[faces_start:faces_end]
    hc_start = page.index("The hyper-connection kernels take the layers")
    hc_end = page.index("\n\n", hc_start)
    hc_para = " ".join(page[hc_start:hc_end].split())
    assert "QuantFormat.training_sensitive = True" in hc_para
    assert (
        "call `eval()` before the first forward or `hc_fused.rebuild_plans`" in hc_para
    )
    assert "hc_fused.rebuild_plans(model)" in hc_para
    assert "as does the load patch" not in hc_para
    assert hc_para.count("model.eval()") == 2


def test_format_load_rebuilds_plans_after_the_loader_evals(tmp_path):
    """Round 12 HIGH-1 & Round 13 MEDIUM-1: format_load._wrap_loader rebuilds
    qwen4_exp HC plans after the loader calls model.eval() (both direct and
    tuple returns), without walking non-qwen4_exp modules."""
    import json
    from types import SimpleNamespace

    from mlx_vlm.models.qwen4_exp import hc_fused

    from omlx.formats import testing
    from omlx.patches import format_load

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    base = _hc_toy_format(8)

    class _Sensitive(type(base)):
        mode = base.mode + "_loader_eval_sensitive"
        training_sensitive = True

    def _sensitive_module():
        fmt = _Sensitive()
        mod = _module(4)
        mod.input_mix_weight_down = _plugin_projection(fmt, WIDTH, LOWRANK, 601)
        mod.input_mix_weight_up = _plugin_projection(fmt, LOWRANK, WIDTH, 602)
        mod.block_inject_weight = _plugin_projection(fmt, WIDTH, HC, 603)
        mod.train(True)
        mod.__dict__.pop("_omlx_hc_plan", None)
        return mod

    def _make_loader(built, captured, as_tuple):
        def loader(path, **kwargs):
            captured["pre_eval"] = hc_fused.rebuild_plan(built)
            built.eval()
            return (built, "meta") if as_tuple else built

        return loader

    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    warm = mx.zeros((1, 17, WIDTH), mx.bfloat16)
    for as_tuple in (False, True):
        built = _sensitive_module()
        captured = {}
        fake = SimpleNamespace(
            __name__="mlx_vlm.utils",
            load_model=_make_loader(built, captured, as_tuple),
        )
        format_load._wrap_loader(fake, "load_model")
        with testing.verified("qwen4_exp"):
            result = fake.load_model(str(tmp_path))
        loaded = result[0] if as_tuple else result
        assert loaded is built
        assert captured["pre_eval"] is not False
        assert loaded.__dict__["_omlx_hc_plan"] is not captured["pre_eval"]
        assert not loaded.input_mix_weight_down.training
        assert hc_fused.fused_forward(loaded, x) is not None
        assert hc_fused.prefill_forward(loaded, warm) is not None

    # A non-qwen4_exp module (even one with a prefix-similar module name) is
    # never walked by the post-load HC plan rebuild.
    class _Other(nn.Module):
        def named_modules(self):
            raise AssertionError("non-qwen4_exp module was walked")

    _Other.__module__ = "mlx_vlm.models.qwen4_exp_other"
    other = _Other()
    fake_other = SimpleNamespace(
        __name__="mlx_vlm.utils", load_model=lambda path, **kwargs: other
    )
    format_load._wrap_loader(fake_other, "load_model")
    with testing.verified("qwen4_exp"):
        assert fake_other.load_model(str(tmp_path)) is other


def test_a_load_builds_each_hc_plan_once(tmp_path, monkeypatch):
    """Land LOW: a loader load builds each HC plan once, at the post-eval
    point, without touching format_load._PENDING (wrapped on a standalone
    namespace, as the loader tests do)."""
    import json
    from types import SimpleNamespace

    from mlx.utils import tree_flatten
    from mlx_vlm.models.qwen4_exp import hc_fused

    from omlx.formats import testing
    from omlx.patches import format_load

    built = _module(4)
    tensors = list(tree_flatten(built.parameters()))
    builds = []
    real_build = hc_fused._build_plan

    def counted(mod):
        builds.append(mod)
        return real_build(mod)

    def loader(path, **kwargs):
        built.__dict__.pop("_omlx_hc_plan", None)
        built.load_weights(tensors, strict=True)
        built.eval()
        return built

    monkeypatch.setattr(hc_fused, "_build_plan", counted)
    fake = SimpleNamespace(__name__="mlx_vlm.utils", load_model=loader)
    format_load._wrap_loader(fake, "load_model")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    with testing.verified("qwen4_exp"):
        loaded = fake.load_model(str(tmp_path))
    assert loaded is built
    assert len(builds) == 1, len(builds)


def test_a_direct_partial_load_leaves_the_plan_stale_until_rebuilt():
    """Land LOW contract: a load not going through the loader's post-eval hook
    leaves an existing plan stale (fail closed); rebuild_plans is explicit."""
    from mlx.utils import tree_flatten
    from mlx_vlm.models.qwen4_exp import hc_fused

    model, _ = _stack(29)
    model.load_weights(list(tree_flatten(model.parameters())), strict=True)
    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    attn_hc = model.model.layers[0].attn_hyper_connection
    assert hc_fused.fused_forward(attn_hc, x) is not None
    new_w = (attn_hc.hc_norm.weight + 0.25).astype(mx.bfloat16)
    model.model.layers[0].load_weights(
        [("attn_hyper_connection.hc_norm.weight", new_w)], strict=False
    )
    with pytest.raises(ValueError, match="rebuild_plan"):
        hc_fused.fused_forward(attn_hc, x)
    hc_fused.rebuild_plans(model)
    assert hc_fused.fused_forward(attn_hc, x) is not None


def test_every_recorded_plan_read_fails_closed():
    """Round 7 HIGH-1: the plan stores every value its build read; mutating
    EACH recorded value in turn (generated from the record itself) fails
    closed with the rebuild_plan error, so a newly read value is covered
    automatically."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    plan = hc_fused.rebuild_plan(module)
    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    warm = mx.zeros((1, 17, WIDTH), mx.bfloat16)
    assert hc_fused.fused_forward(module, x) is not None
    assert plan.reads
    checked = 0
    for owner, kind, name, value in plan.reads:
        if kind == "member":
            if value:
                saved = owner[name]
                owner.pop(name)
                try:
                    with pytest.raises(ValueError, match="rebuild_plan"):
                        hc_fused.fused_forward(module, x)
                finally:
                    owner[name] = saved
            else:
                if name == "bias":
                    added = mx.zeros((owner.output_dims,), mx.bfloat16)
                else:
                    added = _module(4).input_mix_weight_down
                owner[name] = added
                try:
                    with pytest.raises(ValueError, match="rebuild_plan"):
                        hc_fused.fused_forward(module, x)
                finally:
                    owner.pop(name)
            checked += 1
            continue
        if kind == "item":
            old = owner[name]
            owner[name] = mx.zeros_like(old)
            try:
                with pytest.raises(ValueError, match="rebuild_plan"):
                    hc_fused.fused_forward(module, x)
            finally:
                owner[name] = old
            checked += 1
            continue
        old = getattr(owner, name)
        if name == "__class__":
            replacement = type("Sub", (old,), {})
        elif type(old) is bool:
            replacement = not old
        elif type(old) is int:
            replacement = old + 1
        elif type(old) is float:
            replacement = old + 1.0
        elif type(old) is str:
            replacement = old + "_other"
        elif isinstance(old, mx.array):
            replacement = mx.zeros_like(old)
        else:
            replacement = object()
        setattr(owner, name, replacement)
        try:
            with pytest.raises(ValueError, match="rebuild_plan"):
                hc_fused.fused_forward(module, x)
            if name == "input_mix_weight_down":
                with pytest.raises(ValueError, match="rebuild_plan"):
                    hc_fused.prefill_forward(module, warm)
        finally:
            setattr(owner, name, old)
        checked += 1
    assert checked == len(plan.reads)
    down_reads = {
        name
        for owner, _, name, _ in plan.reads
        if owner is module.input_mix_weight_down
    }
    assert {
        "__class__",
        "bias",
        "input_dims",
        "output_dims",
        "mode",
        "bits",
        "group_size",
        "fmt",
        "_tensors",
    } <= down_reads
    norm_reads = {name for owner, _, name, _ in plan.reads if owner is module.hc_norm}
    assert {"__class__", "weight", "group_size", "eps"} <= norm_reads
    module_reads = {name for owner, _, name, _ in plan.reads if owner is module}
    assert {
        "__class__",
        "input_inject_weight",
        "hc_count",
        "hidden_size",
        "hc_lowrank",
        "hc_norm",
        "input_mix_weight_down",
        "input_mix_weight_up",
        "block_inject_weight",
    } <= module_reads


def test_projection_metadata_mutation_fails_closed():
    """Round 8 MEDIUM-1: mutating FormatLinear metadata on a planned projection
    fails closed with ValueError naming rebuild_plan instead of silently running
    the stale kernel."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    hc_fused.rebuild_plan(module)
    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    warm = mx.zeros((1, 17, WIDTH), mx.bfloat16)
    assert hc_fused.fused_forward(module, x) is not None
    assert hc_fused.prefill_forward(module, warm) is not None

    down = module.input_mix_weight_down
    mutations = (
        ("bits", 8),
        ("group_size", 32),
        ("mode", "other"),
        ("input_dims", 64),
        ("output_dims", 64),
        ("fmt", object()),
        ("_tensors", ("other",)),
    )
    for attr, bad in mutations:
        old = getattr(down, attr)
        setattr(down, attr, bad)
        try:
            with pytest.raises(ValueError, match="rebuild_plan"):
                hc_fused.fused_forward(module, x)
            with pytest.raises(ValueError, match="rebuild_plan"):
                hc_fused.prefill_forward(module, warm)
        finally:
            setattr(down, attr, old)
    assert hc_fused.fused_forward(module, x) is not None


def test_eps_validation_and_scalar_type_check():
    """Round 8 MEDIUM-2: non-finite/None/bool eps declines cleanly at plan
    build; post-load scalar type mismatches (bool for float, mx.array) fail
    closed with ValueError naming rebuild_plan."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    warm = mx.zeros((1, 17, WIDTH), mx.bfloat16)
    for bad_eps in (None, True, False, float("nan"), float("inf"), float("-inf")):
        bad = _module(4)
        bad.hc_norm.eps = bad_eps
        assert not hc_fused.compatible(bad, x)
        assert hc_fused.fused_forward(bad, x) is None
        assert hc_fused.prefill_forward(bad, warm) is None
        assert bad.__dict__["_omlx_hc_plan"] is False

    module = _module(4)
    module.hc_norm.eps = 0.0
    plan = hc_fused.rebuild_plan(module)
    assert plan is not False
    assert hc_fused.fused_forward(module, x) is not None
    for mutated in (False, mx.array([0.0, 0.0])):
        module.hc_norm.eps = mutated
        try:
            with pytest.raises(ValueError, match="rebuild_plan"):
                hc_fused.fused_forward(module, x)
            with pytest.raises(ValueError, match="rebuild_plan"):
                hc_fused.prefill_forward(module, warm)
        finally:
            module.hc_norm.eps = 0.0
    assert hc_fused.fused_forward(module, x) is not None


def test_non_container_projection_declines_cleanly():
    """Round 8 LOW-1: a non-container projection object declines cleanly
    instead of crashing with TypeError on membership checks."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    module.input_mix_weight_down = object()
    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    warm = mx.zeros((1, 17, WIDTH), mx.bfloat16)
    assert not hc_fused.compatible(module, x)
    assert hc_fused.fused_forward(module, x) is None
    assert hc_fused.prefill_forward(module, warm) is None
    assert module.__dict__["_omlx_hc_plan"] is False


def test_preplanned_forward_returns_none_without_metal_or_on_cpu(monkeypatch):
    """Round 7 MEDIUM-1: a plan built while Metal was available still declines
    when Metal is gone or the default device is the CPU."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    hc_fused.rebuild_plan(module)
    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    warm = mx.zeros((1, 17, WIDTH), mx.bfloat16)
    with monkeypatch.context() as patch:
        patch.setattr(mx.metal, "is_available", lambda: False)
        assert not hc_fused.compatible(module, x)
        assert not hc_fused.prefill_compatible(module, warm)
        assert hc_fused.fused_forward(module, x) is None
        assert hc_fused.prefill_forward(module, warm) is None
    with monkeypatch.context() as patch:
        patch.setattr(mx, "default_device", lambda: mx.cpu)
        assert not hc_fused.compatible(module, x)
        assert hc_fused.fused_forward(module, x) is None
        assert hc_fused.prefill_forward(module, warm) is None


def test_warm_forward_does_not_rebuild_matrix_readers(monkeypatch):
    """Round 8 NIT-1 + OPT-1: dead _matrix_reader/_make_getter helpers are
    gone, and warm forwards do not re-read tensors or recompile checks."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    assert not hasattr(hc_fused, "_matrix_reader")
    assert not hasattr(hc_fused, "_make_getter")
    module = _module(4)
    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    hc_fused.rebuild_plan(module)
    calls = []
    real_read = hc_fused._read_tensors
    real_compile = hc_fused._compile_plan_checks

    def counted_read(*args, **kwargs):
        calls.append("read")
        return real_read(*args, **kwargs)

    def counted_compile(*args, **kwargs):
        calls.append("compile")
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(hc_fused, "_read_tensors", counted_read)
    monkeypatch.setattr(hc_fused, "_compile_plan_checks", counted_compile)
    hc_fused.fused_forward(module, x)
    hc_fused.fused_forward(module, x)
    hc_fused.prefill_forward(module, mx.zeros((1, 17, WIDTH), mx.bfloat16))
    assert not calls


def test_rebuild_plans_skips_placeholders_on_any_projection_and_dedups_aliases(
    monkeypatch,
):
    """Placeholders on down, up, or inject all clear stale plans and skip
    without raising inside rebuild_plans; aliased HC submodules build once."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    from omlx.formats import linear as formats_linear
    from omlx.formats import testing

    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    for attr in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
        layer = _module(4)
        assert hc_fused.fused_forward(layer, x) is not None
        assert isinstance(layer.__dict__.get("_omlx_hc_plan"), tuple)
        orig = getattr(layer, attr)
        tensors = {name: orig[name] for name in ("weight", "scales", "biases")}
        with testing.verified("qwen4_exp"):
            placeholder = formats_linear.FormatLinear(
                "affine", 4, 64, orig.input_dims, tensors, placeholders=True
            )
        placeholder.scales = placeholder.scales.astype(mx.bfloat16)
        placeholder.biases = placeholder.biases.astype(mx.bfloat16)
        setattr(layer, attr, placeholder)
        assert hc_fused.rebuild_plans(layer) == 0
        assert "_omlx_hc_plan" not in layer.__dict__

    shared = _module(4)
    holder = nn.Module()
    holder.first = shared
    holder.second = shared
    builds = []
    real_build = hc_fused._build_plan

    def counted_build(mod):
        builds.append(mod)
        return real_build(mod)

    monkeypatch.setattr(hc_fused, "_build_plan", counted_build)
    assert hc_fused.rebuild_plans(holder) == 1
    assert len(builds) == 1


def test_a_post_load_array_swap_is_refused_until_rebuild_plan():
    """The immutable plan fails closed when a projection array or the layer
    itself changed after the build; rebuild_plan is the documented way."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    warm = mx.zeros((1, 17, WIDTH), mx.bfloat16)
    assert hc_fused.fused_forward(module, x) is not None
    down = module.input_mix_weight_down
    down["scales"] = down["scales"] * 1
    with pytest.raises(ValueError, match="rebuild_plan"):
        hc_fused.fused_forward(module, x)
    with pytest.raises(ValueError, match="rebuild_plan"):
        hc_fused.prefill_forward(module, warm)
    hc_fused.rebuild_plan(module)
    assert hc_fused.fused_forward(module, x) is not None

    module2 = _module(4)
    assert hc_fused.fused_forward(module2, x) is not None
    module2.input_mix_weight_down = _module(4).input_mix_weight_down
    with pytest.raises(ValueError, match="rebuild_plan"):
        hc_fused.fused_forward(module2, x)


def test_prefill_forward_returns_none_without_metal(monkeypatch):
    """Round 6 MEDIUM-1 + round 8: with no Metal, _build_plan marks the module
    ineligible (False) and neither entry point touches a kernel."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    monkeypatch.setattr(hc_fused, "_EPS_CACHE", {})
    assert hc_fused.fused_forward(module, mx.zeros((1, 2, WIDTH), mx.bfloat16)) is None
    assert hc_fused._EPS_CACHE == {}
    assert (
        hc_fused.prefill_forward(module, mx.zeros((1, 17, WIDTH), mx.bfloat16)) is None
    )
    assert module.__dict__["_omlx_hc_plan"] is False


def test_plan_shares_eps_and_builds_per_plan_tail_launch():
    """LOW-1: equal-eps layers share one evaluated eps array; each plan builds
    its own tail-inject launch (per-plan recording state); an FP32 plan builds
    no tail launch."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    p1 = hc_fused.rebuild_plan(_module(4))
    p2 = hc_fused.rebuild_plan(_module(4))
    assert p1 is not False and p2 is not False
    assert p1.eps is p2.eps
    assert p1.tail_launch is not None and p2.tail_launch is not None
    assert p1.tail_launch is not p2.tail_launch
    fp32 = hc_fused.rebuild_plan(_module(4, dtype=mx.float32))
    assert fp32 is not False and fp32.tail_launch is None


def test_compatible_refuses_zero_rows_and_wrong_width():
    """Round 6 mutants: no rows and a wrong width are refused up front."""
    from mlx_vlm.models.qwen4_exp import hc_fused

    module = _module(4)
    assert not hc_fused.compatible(module, mx.zeros((0, 1, WIDTH), mx.bfloat16))
    assert not hc_fused.compatible(module, mx.zeros((1, 1, WIDTH + 64), mx.bfloat16))
    assert not hc_fused.prefill_compatible(
        module, mx.zeros((1, 17, WIDTH + 64), mx.bfloat16)
    )


def test_cast_bf16_params_to_fp16_rebuilds_hc_plans():
    """Round 6 HIGH-1: the activation-fp16 recast is a weight change; the HC
    plans are rebuilt for the new dtype instead of silently disabling."""
    from mlx.utils import tree_flatten
    from mlx_vlm.models.qwen4_exp import hc_fused

    from omlx.utils.model_loading import cast_bf16_params_to_fp16

    model, _ = _stack(17)
    model.load_weights(list(tree_flatten(model.parameters())), strict=True)
    hc_layers = [
        sub
        for _, sub in model.named_modules()
        if hasattr(sub, "hc_norm") and hasattr(sub, "hc_lowrank")
    ]
    layer = hc_layers[0]
    bf16_x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    assert hc_fused.compatible(layer, bf16_x)
    count = cast_bf16_params_to_fp16(model)
    assert count
    fp16_x = mx.zeros((1, 2, WIDTH), mx.float16)
    assert layer.__dict__["_omlx_hc_plan"].dtype == mx.float16
    assert hc_fused.compatible(layer, fp16_x)
    assert hc_fused.fused_forward(layer, fp16_x) is not None


def test_apply_adapter_rebuilds_hc_plans_for_wrapped_hc_projection(tmp_path):
    """Round 6 HIGH-2: wrapping an HC projection with LoRA must retire the
    stale plan so the composed path runs the wrapped layer."""
    import json

    from mlx.utils import tree_flatten
    from mlx_vlm.models.qwen4_exp import hc_fused

    from omlx.lora_adapters import LoRALinear, apply_adapter

    model, _ = _stack(19)
    model.load_weights(list(tree_flatten(model.parameters())), strict=True)
    hc = model.layers[0].attn_hyper_connection
    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    before = hc_fused.fused_forward(hc, x)
    assert before is not None
    down = hc.input_mix_weight_down
    rank = 2
    a = mx.random.normal((rank, down.input_dims)).astype(mx.bfloat16)
    b = mx.random.normal((down.output_dims, rank)).astype(mx.bfloat16)
    key = (
        "base_model.model.model.layers.0.attn_hyper_connection" ".input_mix_weight_down"
    )
    path = tmp_path / "a"
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": rank, "lora_alpha": 8})
    )
    mx.save_safetensors(
        str(path / "adapter_model.safetensors"),
        {f"{key}.lora_A.weight": a, f"{key}.lora_B.weight": b},
    )
    assert apply_adapter(model, path) == 1
    assert isinstance(hc.input_mix_weight_down, LoRALinear)
    assert not hc_fused.compatible(hc, x)
    got = hc(x)
    want = hc._forward(x)
    assert len(got) == len(want)
    for a, b in zip(got, want):
        assert mx.array_equal(a, b).item()


def test_load_weights_builds_no_plans_until_the_forward(monkeypatch):
    """Land LOW: the vendored load_weights builds no plans (the loader's
    post-eval hook or the first forward builds each once); a direct partial
    load leaves an existing plan stale until rebuild_plan."""
    from mlx.utils import tree_flatten
    from mlx_vlm.models.qwen4_exp import hc_fused
    from mlx_vlm.models.qwen4_exp import qwen4_exp as model_module

    from omlx.formats import linear as formats_linear
    from omlx.formats import testing

    model, head = _stack(13)
    for owner in (model, head):
        for _, sub in owner.named_modules():
            sub.__dict__.pop("_omlx_hc_plan", None)

    builds = []
    real_build = hc_fused._build_plan

    def counted_build(mod):
        builds.append(mod)
        return real_build(mod)

    monkeypatch.setattr(hc_fused, "_build_plan", counted_build)

    lm_hc = [
        sub
        for _, sub in model.named_modules()
        if hasattr(sub, "hc_norm") and hasattr(sub, "hc_lowrank")
    ]
    assert len(lm_hc) == 9  # 4 layers * 2 + final mixer
    model.load_weights(list(tree_flatten(model.parameters())), strict=True)
    assert not builds
    assert all("_omlx_hc_plan" not in sub.__dict__ for sub in lm_hc)

    x = mx.zeros((1, 2, WIDTH), mx.bfloat16)
    for sub in lm_hc:
        assert hc_fused.compatible(sub, x)
        assert hc_fused.fused_forward(sub, x) is not None
    assert len(builds) == len(lm_hc)
    for sub in lm_hc:
        assert hc_fused.fused_forward(sub, x) is not None
    assert len(builds) == len(lm_hc)

    head_hc = [
        sub
        for _, sub in head.named_modules()
        if hasattr(sub, "hc_norm") and hasattr(sub, "hc_lowrank")
    ]
    assert head_hc
    head.load_weights(list(tree_flatten(head.parameters())), strict=True)
    assert len(builds) == len(lm_hc)
    for sub in head_hc:
        assert hc_fused.fused_forward(sub, x) is not None
    assert len(builds) == len(lm_hc) + len(head_hc)

    import mlx.nn as nn

    vlm = model_module.Model.__new__(model_module.Model)
    nn.Module.__init__(vlm)
    vlm.language_model = model
    vlm.mtp = head
    before_vlm = len(builds)
    vlm.load_weights([], strict=False)
    assert len(builds) == before_vlm

    # A direct partial load does not rebuild: the plan is stale (fail closed)
    # until rebuild_plan is called.
    layer = _module(4)
    old_out = hc_fused.fused_forward(layer, x)
    assert old_out is not None
    mx.eval(old_out)
    new_norm = (layer.hc_norm.weight + 0.5).astype(mx.bfloat16)
    layer.load_weights([("hc_norm.weight", new_norm)], strict=False)
    with pytest.raises(ValueError, match="rebuild_plan"):
        hc_fused.fused_forward(layer, x)
    assert hc_fused.rebuild_plan(layer) is not False
    assert hc_fused.fused_forward(layer, x) is not None

    # An unfilled placeholder under a direct load keeps the plan stale; the
    # explicit rebuild raises format_load's placeholder error.
    down = layer.input_mix_weight_down
    tensors = {name: down[name] for name in ("weight", "scales", "biases")}
    with testing.verified("qwen4_exp"):
        placeholder = formats_linear.FormatLinear(
            "affine", 4, 64, WIDTH, tensors, placeholders=True
        )
    placeholder.scales = placeholder.scales.astype(mx.bfloat16)
    placeholder.biases = placeholder.biases.astype(mx.bfloat16)
    layer.input_mix_weight_down = placeholder
    with pytest.raises(ValueError, match="rebuild_plan"):
        hc_fused.fused_forward(layer, x)
    with pytest.raises(ValueError, match="placeholder"):
        hc_fused.rebuild_plan(layer)
