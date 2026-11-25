# SPDX-License-Identifier: Apache-2.0
"""End-to-end acceptance gate for real Qwen3.8-Flash-Next oQh4 + FP8 PLE.

Skips cleanly when the local TB5 checkpoint is absent (no environment variables;
see project rule 30). When present, verifies:
1. Coverage: all 144 trunk routed switch_mlp.{gate,up,down}_proj layers load as
   FormatSwitchLinear(OqhAffineFormat) (4-bit and reinvested 5-bit, g=128),
   trunk affine layers load as FormatLinear(AffineFormat), mtp.* stays plain
   affine, hc_fused plans and transformed routed-decode plans are active, and
   FP8 PLE loads intact.
2. Numerics contract (Rule 33 + Amendments 1 & 2): C4e verify-vs-decode KL and
   top-1 within cross-chip bars (KL <= 6.757e-3 code / 2.004e-2 prose,
   top-1 >= 0.9355), zero NaN/Inf, and C5 same-chip bitwise repeatability.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.formats import kernels
from omlx.formats.affine import AffineFormat
from omlx.formats.linear import FormatLinear, FormatSwitchLinear
from omlx.formats.oqh import OqhAffineFormat
from omlx.patches import (
    m5_gather_qmm,
    qwen35_gdn_prework,
    qwen35_moe_gate_up,
    qwen35_moe_router,
    qwen35_moe_weighted_sum,
)
from omlx.patches import qwen35_moe_routed_decode as routed

FLASH_NEXT_OQH4_PATH = Path("/Volumes/TB5/llm/Qwen3.8-Flash-Next-oQh4-fp8ple")

# Amendment 2 C4e bars (1.5x cross-chip MLX floor on Qwen3.8-Flash-Next).
_C4E_KL_CODE = 6.757e-3
_C4E_KL_PROSE = 2.004e-2
_C4E_TOP1 = 0.9355

pytestmark = [
    pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal"),
    pytest.mark.skipif(
        not FLASH_NEXT_OQH4_PATH.is_dir(),
        reason=f"real Flash-Next oQh4 checkpoint not present at {FLASH_NEXT_OQH4_PATH}",
    ),
]


def _kl_and_top1(logits_p: mx.array, logits_q: mx.array) -> tuple[float, float]:
    """Mean KL(P || Q) and top-1 agreement in FP32 over [..., V] logits."""
    p = logits_p.astype(mx.float32).reshape(-1, logits_p.shape[-1])
    q = logits_q.astype(mx.float32).reshape(-1, logits_q.shape[-1])
    lp = p - mx.logsumexp(p, axis=-1, keepdims=True)
    lq = q - mx.logsumexp(q, axis=-1, keepdims=True)
    kl = mx.sum(mx.exp(lp) * (lp - lq), axis=-1)
    top1 = mx.mean((mx.argmax(p, axis=-1) == mx.argmax(q, axis=-1)).astype(mx.float32))
    mx.eval(kl, top1)
    return float(mx.mean(kl).item()), float(top1.item())


def _load_flash_next(path: Path = FLASH_NEXT_OQH4_PATH):
    from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
    from omlx.model_settings import ModelSettings
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    # Must run before importing mlx_vlm.models.qwen4_exp so the vendored compat
    # package is installed in sys.modules first.
    maybe_apply_pre_load_patches(
        str(path),
        model_settings=ModelSettings(
            qwen4_ple_ssd_offload=True, mtp_enabled=True, a8_prefill_enabled=False
        ),
        for_vlm=True,
    )
    from mlx_vlm.models.qwen4_exp import hc_fused
    from mlx_vlm.utils import load as vlm_load

    with _force_qwen4_exp_sanitize_on_load(path):
        model, processor = vlm_load(str(path), trust_remote_code=False, lazy=True)
    mx.eval(model.parameters())

    hc_plans = hc_fused.rebuild_plans(model)
    m5_gather_qmm.apply_m5_gather_qmm_workaround()
    from mlx_vlm.models.qwen3_5 import language as q35
    from mlx_vlm.models.qwen3_5.speculative_verifier import (
        Qwen3_5BatchInvariantForward,
    )

    gdn_cls = q35.Qwen3_5GatedDeltaNet
    gdn_call = getattr(gdn_cls.__call__, "_omlx_gdn_prework_original", gdn_cls.__call__)
    gdn_verify = getattr(
        Qwen3_5BatchInvariantForward._gated_delta,
        "_omlx_gdn_prework_original",
        Qwen3_5BatchInvariantForward._gated_delta,
    )
    restore = (gdn_cls, gdn_call, Qwen3_5BatchInvariantForward, gdn_verify)
    qwen35_gdn_prework.apply_qwen35_gdn_prework_patch()
    qwen35_moe_router.apply_qwen35_moe_router_patch()
    qwen35_moe_weighted_sum.apply_qwen35_moe_weighted_sum_patch()
    routed.apply_qwen35_moe_routed_decode_patch()
    pre_fusion_oqh = [
        (name, m.bits, m.group_size)
        for name, m in model.named_modules()
        if isinstance(m, FormatSwitchLinear) and isinstance(m.fmt, OqhAffineFormat)
    ]
    fused_count = qwen35_moe_gate_up.apply_qwen35_moe_gate_up_fusion(model)
    routed.warm_moe_model(model, mtp_enabled=True, warm_dispatch=False)
    routed.warm_engine_dispatch(model, None, mtp_enabled=True)
    return model, processor, pre_fusion_oqh, fused_count, hc_plans, restore


@pytest.fixture(scope="module")
def flash_next_oqh4():
    (
        model,
        processor,
        pre_fusion_oqh,
        fused_count,
        hc_plans,
        restore,
    ) = _load_flash_next(FLASH_NEXT_OQH4_PATH)
    lm = getattr(model, "language_model", model)
    layers = getattr(getattr(lm, "model", lm), "layers", [])
    embeddings = [
        layer.ple.ple_embedding.ngram_embedding
        for layer in layers
        if getattr(layer, "ple", None) is not None
    ]
    failed = False
    try:
        yield model, processor, pre_fusion_oqh, fused_count, hc_plans
    except BaseException:
        failed = True
        raise
    finally:
        close = getattr(model, "close", None)
        if close is not None:
            close()
        gdn_cls, gdn_call, verifier, gdn_verify = restore
        gdn_cls.__call__ = gdn_call
        verifier._gated_delta = gdn_verify
        if "_omlx_gdn_prework_patched" in gdn_cls.__dict__:
            delattr(gdn_cls, "_omlx_gdn_prework_patched")
        qwen35_gdn_prework._PATCHED = False
        del model, processor, lm, layers
        mx.clear_cache()
        if not failed:
            assert embeddings and all(
                getattr(e, "_prefetch_closed", False) for e in embeddings
            )


def test_flash_next_oqh4_coverage_and_plans(flash_next_oqh4):
    model, _, pre_fusion_oqh, fused_count, hc_plans = flash_next_oqh4
    cfg = json.loads((FLASH_NEXT_OQH4_PATH / "config.json").read_text())
    oqh_cfg = cfg["omlx_oqh"]
    assert oqh_cfg["version"] == 1
    assert len(oqh_cfg["tensors"]) == 144
    assert all("mtp" not in k for k in oqh_cfg["tensors"])

    # Before gate+up fusion: 48 layers x 3 projections = 144 oqh_affine stacks,
    # each matching the checkpoint's exact recorded bits and group_size=128.
    assert len(pre_fusion_oqh) == 144
    assert all("mtp" not in n for n, _, _ in pre_fusion_oqh)
    expected_5bit_down = sum(
        1
        for k, meta in oqh_cfg["tensors"].items()
        if k.endswith(".down_proj") and meta["bits"] == 5
    )
    assert expected_5bit_down > 0
    for name, bits, gs in pre_fusion_oqh:
        meta = oqh_cfg["tensors"][name]
        assert bits == meta["bits"]
        assert gs == meta["group_size"] == 128

    # After gate+up fusion: 48 fused gate_up_proj + 48 down_proj + 48 kept
    # gate_proj + 48 kept up_proj = 192 FormatSwitchLinear(OqhAffineFormat).
    oqh_switches = [
        (name, m)
        for name, m in model.named_modules()
        if isinstance(m, FormatSwitchLinear) and isinstance(m.fmt, OqhAffineFormat)
    ]
    assert len(oqh_switches) == 192
    assert all("mtp" not in n for n, _ in oqh_switches)
    assert all(m.group_size == 128 and m.bits in (4, 5) for _, m in oqh_switches)
    assert sum(1 for n, _ in oqh_switches if n.endswith(".gate_up_proj")) == 48
    assert sum(1 for n, _ in oqh_switches if n.endswith(".down_proj")) == 48
    expected_b5 = {
        k
        for k, meta in oqh_cfg["tensors"].items()
        if k.endswith(".down_proj") and meta["bits"] == 5
    }
    actual_b5 = {n for n, m in oqh_switches if n.endswith(".down_proj") and m.bits == 5}
    assert len(expected_b5) > 0 and actual_b5 == expected_b5
    assert fused_count >= 48

    # Trunk and MTP affine projections load as FormatLinear(AffineFormat) (zero
    # legacy nn.QuantizedLinear), and FP8 PLE attaches to layers.
    legacy_ql = [
        name for name, m in model.named_modules() if isinstance(m, nn.QuantizedLinear)
    ]
    assert not legacy_ql, f"unexpected legacy nn.QuantizedLinear: {legacy_ql[:5]}"
    named_affine = [
        (name, m)
        for name, m in model.named_modules()
        if isinstance(m, FormatLinear) and isinstance(m.fmt, AffineFormat)
    ]
    assert len(named_affine) > 400
    mtp_format = [
        (name, m)
        for name, m in model.named_modules()
        if "mtp" in name and isinstance(m, (FormatLinear, FormatSwitchLinear))
    ]
    assert len(mtp_format) > 1 and all(
        isinstance(m.fmt, AffineFormat) for _, m in mtp_format
    )

    # All 48 trunk MoE blocks build folded transformed routed-decode plans
    # and every configured FP8 PLE layer is attached and open.
    lm = getattr(model, "language_model", model)
    layers = getattr(getattr(lm, "model", lm), "layers", [])
    assert len(layers) == 48
    ple_ids = tuple(cfg.get("text_config", cfg).get("ple_layer_ids", ()))
    assert ple_ids
    # config.ple_layer_ids is 1-indexed (see Qwen4ExpDecoderLayer.__init__).
    actual_ple_ids = tuple(
        idx + 1
        for idx, layer in enumerate(layers)
        if getattr(layer, "ple", None) is not None
    )
    assert actual_ple_ids == ple_ids
    embeddings = [layers[idx - 1].ple.ple_embedding.ngram_embedding for idx in ple_ids]
    assert not any(getattr(e, "_prefetch_closed", True) for e in embeddings)

    # Every trunk hyper-connection has a built (truthy, not ``False``) plan.
    from mlx_vlm.models.qwen4_exp import hc_fused

    trunk_hc = [
        hc
        for layer in layers
        for hc in (layer.attn_hyper_connection, layer.mlp_hyper_connection)
    ]
    mixer = getattr(getattr(lm, "model", lm), "hyper_connection_mixer", None)
    if mixer is not None:
        trunk_hc.append(mixer)
    assert hc_plans >= len(trunk_hc)
    assert all(bool(hc.__dict__.get(hc_fused._PLAN_ATTR)) for hc in trunk_hc)
    for idx, layer in enumerate(layers):
        plan = routed._cached_plan(layer.mlp, mx.bfloat16)
        assert plan is not None, f"layer {idx}: no routed-decode plan"
        assert plan.launches.fold is True
        assert plan.launches.routed_x is True
        assert plan.launches.split_out is True
        assert plan.launches.split_in is True
        assert plan.launches.shared_uses_rx is False
        assert plan.window is not None
        # Warmed at load before any forward pass runs.
        sig1 = ((1, 1, plan.launches.hidden), mx.bfloat16, plan.experts)
        assert sig1 in plan.proof.graphs, f"layer {idx}: decode graph not warmed"
        for win_rows in (2, 4, 8):
            wsig = (
                (1, win_rows, plan.launches.hidden),
                mx.bfloat16,
                plan.experts,
                plan.quantized_router,
            )
            assert (
                wsig in plan.window.graphs
            ), f"layer {idx}: window {win_rows} graph not warmed"


def test_flash_next_oqh4_verify_vs_decode_kl_and_repeatability(flash_next_oqh4):
    from omlx.patches.qwen35_verify_qmm import set_verify_qmm_armed

    model, processor, _, _, _ = flash_next_oqh4
    proofs_before = kernels.proof_count()
    lm = getattr(model, "language_model", model)
    tok = processor.tokenizer
    code_ids = tok.encode(
        "def binary_search(arr, target):\n"
        "    lo, hi = 0, len(arr) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n"
    )
    prose_ids = tok.encode(
        "The history of astronomy spans thousands of years, from ancient sky "
        "watchers charting the planets to modern space telescopes observing "
        "the earliest galaxies in the observable universe.\n"
    )
    code_prompt = mx.array([code_ids[:-8]])
    code_tail = mx.array([code_ids[-8:]])
    prose_prompt = mx.array([prose_ids[:-8]])
    prose_tail = mx.array([prose_ids[-8:]])

    def run_serial(seq_prompt, seq_tail):
        cache = lm.make_cache()
        p_out = lm(seq_prompt, cache=cache)
        mx.eval(p_out.logits)
        outs = []
        for i in range(seq_tail.shape[1]):
            out = lm(seq_tail[:, i : i + 1], cache=cache)
            logits = out.logits if hasattr(out, "logits") else out
            mx.eval(logits)
            outs.append(logits)
        return mx.concatenate(outs, axis=1)

    def run_verify(seq_prompt, seq_tail, chunk: int):
        cache = lm.make_cache()
        p_out = lm(seq_prompt, cache=cache)
        mx.eval(p_out.logits)
        outs = []
        for i in range(0, seq_tail.shape[1], chunk):
            step = seq_tail[:, i : i + chunk]
            set_verify_qmm_armed(True, row_exact=False)
            try:
                out = lm(step, cache=cache, return_hidden=True)
                logits = out.logits if hasattr(out, "logits") else out
                mx.eval(logits)
            finally:
                set_verify_qmm_armed(False)
            assert (
                out.gdn_states is not None
            ), "verify path did not open speculative cache"
            out.gdn_states.commit([step.shape[1]])
            outs.append(logits)
        return mx.concatenate(outs, axis=1)

    for label, seq_prompt, seq_tail, kl_bar in (
        ("code", code_prompt, code_tail, _C4E_KL_CODE),
        ("prose", prose_prompt, prose_tail, _C4E_KL_PROSE),
    ):
        serial_a = run_serial(seq_prompt, seq_tail)
        serial_b = run_serial(seq_prompt, seq_tail)
        assert mx.isfinite(serial_a).all().item()
        # C5 same-chip bitwise repeatability on serial decode.
        assert mx.array_equal(serial_a, serial_b).item()

        for window in (2, 4, 8):
            ver_a = run_verify(seq_prompt, seq_tail, window)
            ver_b = run_verify(seq_prompt, seq_tail, window)
            assert mx.isfinite(ver_a).all().item()
            # C5 same-chip bitwise repeatability on verify windows.
            assert mx.array_equal(ver_a, ver_b).item()
            kl, top1 = _kl_and_top1(ver_a, serial_a)
            assert kl <= kl_bar, f"{label} window={window} KL={kl:.4e} > {kl_bar:.4e}"
            assert (
                top1 >= _C4E_TOP1
            ), f"{label} window={window} top1={top1:.4f} < {_C4E_TOP1}"

    assert (
        kernels.proof_count() == proofs_before
    ), f"cold proofs on first request: {kernels.proof_count() - proofs_before}"
    layers = getattr(getattr(lm, "model", lm), "layers", [])
    for idx, layer in enumerate(layers):
        plan = routed._cached_plan(layer.mlp, mx.bfloat16)
        assert plan is not None, f"layer {idx}: missing routed plan"
        assert (
            not plan.proof.failed and len(plan.proof.graphs) >= 1
        ), f"layer {idx}: routed decode did not engage"
        assert (
            not plan.window.failed and len(plan.window.graphs) >= 3
        ), f"layer {idx}: routed verify window did not engage"
