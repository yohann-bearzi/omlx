# SPDX-License-Identifier: Apache-2.0
"""oQh's MoE routes rotate once: the gate/up fusion, the verify gather and the
prefill weighted sum hand each layer a prepared activation instead of
rotating every sorted pair."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.switch_layers import SwitchGLU

from omlx.formats import linear as linear_mod
from omlx.formats import oqh, testing, transforms
from omlx.formats.linear import FormatLinear, FormatSwitchLinear
from omlx.patches import (
    moe_verify_gather,
    qwen35_moe_gate_up,
    qwen35_moe_weighted_sum,
)
from omlx.patches.moe_routes import sort_routes
from tests.test_formats_oqh import _group_case

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

K, N, E, TOP_K = 640, 128, 8, 2


def _stack(seed=0, k=K, n=N):
    parts = [_group_case(k, n, 4, 128, seed=seed + e) for e in range(E)]
    signs = parts[0][0]["oqh_signs"]
    tensors = {
        "weight": mx.stack([p[0]["weight"] for p in parts]),
        "oqh_scales": mx.stack([p[0]["oqh_scales"] for p in parts]),
        "oqh_biases": mx.stack([p[0]["oqh_biases"] for p in parts]),
        "oqh_signs": signs,
    }
    with testing.verified("example"):
        return FormatSwitchLinear(oqh.OQH_AFFINE, 4, 128, k, tensors)


def _dense_oqh(k, n, bits=4, gs=128, seed=0):
    tensors, _ = _group_case(k, n, bits, gs, seed=seed)
    with testing.verified("example"):
        return FormatLinear(oqh.OQH_AFFINE, bits, gs, k, tensors)


def _glu():
    gate, up = _stack(0), _stack(2)
    down = _stack(4, k=N, n=K)
    glu = SwitchGLU(K, N, E)
    glu.gate_proj, glu.up_proj, glu.down_proj = gate, up, down
    return glu


def _plain_glu():
    """The unfused reference: its own layers, called as a model calls them."""
    gate, up = _stack(0), _stack(2)
    down = _stack(4, k=N, n=K)
    glu = SwitchGLU(K, N, E)
    glu.gate_proj, glu.up_proj, glu.down_proj = gate, up, down
    return glu


def test_the_fused_gate_up_rotates_the_token_rows_once(monkeypatch):
    fused, plain = _glu(), _plain_glu()
    qwen35_moe_gate_up._ensure_call_patch()
    qwen35_moe_gate_up._fuse_one(fused)
    mx.random.seed(1)
    for rows in (1, 4, 64):
        x = mx.random.normal((1, rows, K)).astype(mx.bfloat16)
        indices = mx.random.randint(0, E, (1, rows, TOP_K)).astype(mx.uint32)
        seen = []
        original = transforms.prepare_bound

        def spy(bound, value, _seen=seen, _orig=original):
            _seen.append(tuple(value.shape))
            return _orig(bound, value)

        monkeypatch.setattr(transforms, "prepare_bound", spy)
        got = fused(x, indices)
        monkeypatch.setattr(transforms, "prepare_bound", original)
        want = plain(x, indices)
        assert mx.isfinite(got).all().item() and mx.isfinite(want).all().item()
        assert mx.allclose(
            got.astype(mx.float32), want.astype(mx.float32), atol=6e-2, rtol=6e-2
        ).item()
        # The rotation ran on the token rows, never on the expanded pairs.
        assert seen and seen[0] == (1, rows, K)
        assert (rows * TOP_K, 1, K) not in seen
        if not linear_mod._nax_host():
            assert seen == [(1, rows, K)]


def test_the_verify_gather_keeps_one_row_bits_and_rotates_the_block(monkeypatch):
    fused, plain = _glu(), _plain_glu()
    qwen35_moe_gate_up._fuse_one(fused)
    switch = fused
    mx.random.seed(2)
    for length in (2, 4, 8):
        x = mx.random.normal((1, length, K)).astype(mx.bfloat16)
        indices = mx.random.randint(0, E, (1, length, TOP_K)).astype(mx.uint32)
        seen = []
        original = transforms.prepare_bound

        def spy(bound, value, _seen=seen, _orig=original):
            _seen.append(tuple(value.shape))
            return _orig(bound, value)

        monkeypatch.setattr(transforms, "prepare_bound", spy)
        got = moe_verify_gather.fused_switch(switch, x, indices)
        monkeypatch.setattr(transforms, "prepare_bound", original)
        assert got is not None
        assert seen == [(1, length, K), (length * TOP_K, N)]
        for i in range(length):
            one = plain(x[:, i : i + 1], indices[:, i : i + 1])
            assert mx.allclose(
                got[:, i : i + 1].astype(mx.float32),
                one.astype(mx.float32),
                atol=6e-2,
                rtol=6e-2,
            ).item()


def test_the_prefill_weighted_sum_rotates_before_sorting(monkeypatch):
    fused, plain = _glu(), _plain_glu()
    qwen35_moe_gate_up._fuse_one(fused)
    switch = fused
    rows = 128
    x = mx.random.normal((1, rows, K)).astype(mx.bfloat16)
    inds = mx.random.randint(0, E, (1, rows, TOP_K)).astype(mx.uint32)
    scores = mx.softmax(mx.random.normal((1, rows, TOP_K)), axis=-1)
    seen = []
    original = transforms.prepare_bound

    def spy(bound, value):
        seen.append(tuple(value.shape))
        return original(bound, value)

    monkeypatch.setattr(transforms, "prepare_bound", spy)
    got = qwen35_moe_weighted_sum._native_switch_weighted_sum(
        switch,
        x,
        inds,
        scores,
        lambda sorted_out, inv, s: sorted_out,
    )
    monkeypatch.setattr(transforms, "prepare_bound", original)
    # The token rows rotate once, not the sorted pairs.
    assert seen and seen[0] == (1, rows, K)
    assert (rows * TOP_K, 1, K) not in seen
    if not linear_mod._nax_host():
        assert seen == [(1, rows, K)]
    # Reference: the unfused block over the same routes.
    x_tok, row_map, idx, inv = sort_routes(mx.expand_dims(x, (-2, -3)), inds)
    want = plain.down_proj(
        plain.activation(
            plain.up_proj(x_tok[row_map], idx, sorted_indices=True),
            plain.gate_proj(x_tok[row_map], idx, sorted_indices=True),
        ),
        idx,
        sorted_indices=True,
    )
    assert mx.isfinite(got).all().item() and mx.isfinite(want).all().item()
    a = got.astype(mx.float32)
    b = want.astype(mx.float32)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert mx.allclose(a, b, atol=0.2, rtol=0.2).item()


def test_sorted_transform_routes_use_uint32_and_stay_finite():
    fused = _glu()
    qwen35_moe_gate_up._ensure_call_patch()
    qwen35_moe_gate_up._fuse_one(fused)
    rows = 64
    x = mx.random.normal((1, rows, K)).astype(mx.bfloat16)
    indices = mx.random.randint(0, E, (1, rows, TOP_K)).astype(mx.uint32)
    got = fused(x, indices)
    assert mx.isfinite(got).all().item()
    kept_gu = fused.gate_up_proj.__dict__["_omlx_format_call"]
    assert kept_gu.sorted is not None and kept_gu.sorted.proof.ran


def test_non_transformed_switch_calls_fused_gate_up_activation(monkeypatch):
    glu = SwitchGLU(K, N, E)
    nn.quantize(glu, group_size=64, bits=4)
    glu.eval()
    qwen35_moe_gate_up._ensure_call_patch()
    qwen35_moe_gate_up._fuse_one(glu)
    calls = []

    # 5-arg spy matching test_formats_tile_epilogue.py: rejects unexpected kwargs.
    def observe(proj, x, idx, act, token_rows=None):
        calls.append((proj, token_rows is not None))
        return None

    monkeypatch.setattr(qwen35_moe_gate_up, "fused_gate_up_activation", observe)
    monkeypatch.setattr(qwen35_moe_weighted_sum, "fused_gate_up_activation", observe)
    rows = 64
    x = mx.random.normal((1, rows, K)).astype(mx.bfloat16)
    indices = mx.random.randint(0, E, (1, rows, TOP_K)).astype(mx.uint32)
    scores = mx.softmax(mx.random.normal((1, rows, TOP_K)), axis=-1)
    glu(x, indices)
    assert len(calls) == 1
    qwen35_moe_weighted_sum._native_switch_weighted_sum(
        glu, x, indices, scores, lambda sorted_out, inv, s: sorted_out
    )
    assert len(calls) == 2


def test_verify_gather_declines_before_rotating(monkeypatch):
    fused = _glu()
    qwen35_moe_gate_up._fuse_one(fused)
    seen = []
    original = transforms.prepare_bound
    monkeypatch.setattr(
        transforms,
        "prepare_bound",
        lambda bound, value: seen.append(tuple(value.shape)) or original(bound, value),
    )
    x_int = mx.zeros((1, 2, K), mx.int32)
    idx = mx.zeros((1, 2, TOP_K), mx.uint32)
    assert moe_verify_gather.fused_switch(fused, x_int, idx) is None
    assert seen == []
    x_bf = mx.zeros((1, 2, K), mx.bfloat16)
    assert moe_verify_gather.fused_switch(fused, x_bf, idx[..., :0]) is None
    assert seen == []


def test_verify_gather_binds_transforms_in_plan():
    fused = _glu()
    qwen35_moe_gate_up._fuse_one(fused)
    x = mx.random.normal((1, 2, K)).astype(mx.bfloat16)
    indices = mx.random.randint(0, E, (1, 2, TOP_K)).astype(mx.uint32)
    got = moe_verify_gather.fused_switch(fused, x, indices)
    assert got is not None
    assert fused._omlx_verify_gather_plan[3].gu_tx is not None
    assert fused._omlx_verify_gather_plan[3].dn_tx is not None
    assert "_omlx_verify_gather_tx" not in vars(fused)


def test_unfused_weighted_sum_rotates_token_rows_once(monkeypatch):
    unfused = _glu()
    rows = 64
    x = mx.random.normal((1, rows, K)).astype(mx.bfloat16)
    inds = mx.random.randint(0, E, (1, rows, TOP_K)).astype(mx.uint32)
    scores = mx.softmax(mx.random.normal((1, rows, TOP_K)), axis=-1)
    seen = []
    original = transforms.prepare_bound
    monkeypatch.setattr(
        transforms,
        "prepare_bound",
        lambda bound, value: seen.append(tuple(value.shape)) or original(bound, value),
    )
    got = qwen35_moe_weighted_sum._native_switch_weighted_sum(
        unfused, x, inds, scores, lambda sorted_out, inv, s: sorted_out
    )
    assert mx.isfinite(got).all().item()
    assert seen and seen[0] == (1, rows, K)
    assert (rows * TOP_K, 1, K) not in seen


def test_oqh_experts_skips_transform_cache():
    class _Stub:
        def oqh_rotated_input(self, x):
            return x

        def oqh_experts(self, x_sorted, idx, sorted_indices=True):
            return x_sorted

    rows = 4
    x = mx.ones((1, rows, K), mx.bfloat16)
    inds = mx.zeros((1, rows, TOP_K), mx.uint32)
    scores = mx.ones((1, rows, TOP_K), mx.float32)
    out = qwen35_moe_weighted_sum._native_switch_weighted_sum(
        _Stub(), x, inds, scores, lambda sorted_out, inv, s: sorted_out
    )
    assert out.shape[0] == rows * TOP_K


def test_cached_transforms_precompute_memo_key(monkeypatch):
    fused = _glu()
    qwen35_moe_gate_up._ensure_call_patch()
    qwen35_moe_gate_up._fuse_one(fused)
    x2 = mx.random.normal((1, 2, K)).astype(mx.bfloat16)
    idx2 = mx.random.randint(0, E, (1, 2, TOP_K)).astype(mx.uint32)
    x64 = mx.random.normal((1, 64, K)).astype(mx.bfloat16)
    idx64 = mx.random.randint(0, E, (1, 64, TOP_K)).astype(mx.uint32)
    scores64 = mx.softmax(mx.random.normal((1, 64, TOP_K)), axis=-1)
    # Warm all three cached paths.
    mx.eval(fused(x2, idx2))
    mx.eval(moe_verify_gather.fused_switch(fused, x2, idx2))
    mx.eval(
        qwen35_moe_weighted_sum._native_switch_weighted_sum(
            fused, x64, idx64, scores64, lambda sorted_out, inv, s: sorted_out
        )
    )
    monkeypatch.setattr(
        transforms,
        "key",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("key recomputed")),
    )
    mx.eval(fused(x2, idx2))
    mx.eval(moe_verify_gather.fused_switch(fused, x2, idx2))
    mx.eval(
        qwen35_moe_weighted_sum._native_switch_weighted_sum(
            fused, x64, idx64, scores64, lambda sorted_out, inv, s: sorted_out
        )
    )


def test_fused_gate_up_skips_rotation_on_empty_x(monkeypatch):
    fused, _ = _glu(), _plain_glu()
    qwen35_moe_gate_up._ensure_call_patch()
    qwen35_moe_gate_up._fuse_one(fused)
    seen = []
    original = transforms.prepare_bound
    monkeypatch.setattr(
        transforms,
        "prepare_bound",
        lambda bound, value: seen.append(value.size) or original(bound, value),
    )
    x = mx.zeros((1, 0, K), mx.bfloat16)
    out = fused(x, mx.zeros((1, 0, TOP_K), mx.uint32))
    assert out.shape[:2] == (1, 0)
    assert not seen


def test_weighted_sum_skips_rotation_on_empty_x(monkeypatch):
    fused = _glu()
    qwen35_moe_gate_up._fuse_one(fused)
    unfused = _glu()
    seen = []
    original = transforms.prepare_bound
    monkeypatch.setattr(
        transforms,
        "prepare_bound",
        lambda bound, value: seen.append(value.size) or original(bound, value),
    )
    x = mx.zeros((1, 0, K), mx.bfloat16)
    inds = mx.zeros((1, 0, TOP_K), mx.uint32)
    scores = mx.zeros((1, 0, TOP_K), mx.float32)
    for switch in (fused, unfused):
        out = qwen35_moe_weighted_sum._native_switch_weighted_sum(
            switch, x, inds, scores, lambda sorted_out, inv, s: sorted_out
        )
        assert out.shape[0] == 0
    assert seen == []


@pytest.mark.parametrize("bad_limit", [float("inf"), float("nan")])
def test_non_finite_swiglu_limit_skips_apply_split_swiglu(monkeypatch, bad_limit):
    from omlx.patches import m5_gather_qmm as _m5

    fused = _glu()
    qwen35_moe_gate_up._ensure_call_patch()
    qwen35_moe_gate_up._fuse_one(fused)
    monkeypatch.setattr(_m5, "_swiglu_limit", lambda act: bad_limit)
    monkeypatch.setattr(
        qwen35_moe_gate_up, "fused_gate_up_activation", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        qwen35_moe_weighted_sum, "fused_gate_up_activation", lambda *a, **kw: None
    )
    called = []
    dn_tx, dn_tensors = oqh.OQH_AFFINE.input_transform(fused.down_proj), (
        fused.down_proj["oqh_signs"],
    )
    guarded = transforms.InputTransform(
        version=dn_tx.version,
        kind=dn_tx.kind,
        tensors=dn_tx.tensors,
        block=dn_tx.block,
        apply=dn_tx.apply,
        apply_split_swiglu=lambda *a, **kw: called.append(kw),
        inverse_weight=dn_tx.inverse_weight,
    )
    monkeypatch.setattr(
        qwen35_moe_gate_up,
        "_stack_transforms",
        lambda s: (
            qwen35_moe_gate_up._bind_with_key(s.gate_up_proj),
            (guarded, dn_tensors, transforms.key(guarded, dn_tensors)),
        ),
    )
    monkeypatch.setattr(
        qwen35_moe_weighted_sum,
        "_weighted_sum_transforms",
        lambda s: (
            qwen35_moe_weighted_sum._bind_with_key(s.gate_up_proj),
            (guarded, dn_tensors, transforms.key(guarded, dn_tensors)),
        ),
    )
    x = mx.random.normal((1, 4, K)).astype(mx.bfloat16)
    inds = mx.random.randint(0, E, (1, 4, TOP_K)).astype(mx.uint32)
    scores = mx.softmax(mx.random.normal((1, 4, TOP_K)), axis=-1)
    out_gu = fused(x, inds)
    out_ws = qwen35_moe_weighted_sum._native_switch_weighted_sum(
        fused, x, inds, scores, lambda sorted_out, inv, s: sorted_out
    )
    assert mx.isfinite(out_gu).all().item() and mx.isfinite(out_ws).all().item()
    assert not called


def test_declined_apply_split_swiglu_empty_activation_resets_rotated(monkeypatch):
    from omlx.patches import m5_gather_qmm as _m5

    fused = _glu()
    qwen35_moe_gate_up._ensure_call_patch()
    qwen35_moe_gate_up._fuse_one(fused)
    monkeypatch.setattr(_m5, "_swiglu_limit", lambda act: None)
    monkeypatch.setattr(
        qwen35_moe_gate_up, "fused_gate_up_activation", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        qwen35_moe_weighted_sum, "fused_gate_up_activation", lambda *a, **kw: None
    )
    declined = []
    dn_tx = oqh.OQH_AFFINE.input_transform(fused.down_proj)
    dn_tensors = (fused.down_proj["oqh_signs"],)
    declining = transforms.InputTransform(
        version=dn_tx.version,
        kind=dn_tx.kind,
        tensors=dn_tx.tensors,
        block=dn_tx.block,
        apply=dn_tx.apply,
        apply_split_swiglu=lambda *a, **kw: declined.append(1) or None,
        inverse_weight=dn_tx.inverse_weight,
    )
    monkeypatch.setattr(
        qwen35_moe_gate_up,
        "_stack_transforms",
        lambda s: (
            qwen35_moe_gate_up._bind_with_key(s.gate_up_proj),
            (declining, dn_tensors, transforms.key(declining, dn_tensors)),
        ),
    )
    monkeypatch.setattr(
        qwen35_moe_weighted_sum,
        "_weighted_sum_transforms",
        lambda s: (
            qwen35_moe_weighted_sum._bind_with_key(s.gate_up_proj),
            (declining, dn_tensors, transforms.key(declining, dn_tensors)),
        ),
    )
    fused.activation = lambda u, g: u[:0]

    direct_prepared = []
    via_call = []
    orig_call = FormatSwitchLinear.__call__

    def spy_call(self, x_in, idx_in, sorted_indices=False):
        if self is fused.down_proj:
            via_call.append(tuple(x_in.shape))
            return mx.zeros((*x_in.shape[:-1], K), x_in.dtype)
        return orig_call(self, x_in, idx_in, sorted_indices=sorted_indices)

    def spy_prepared(x_in, idx_in, sorted_indices=False, kept=None):
        direct_prepared.append(tuple(x_in.shape))
        return mx.zeros((*x_in.shape[:-1], K), x_in.dtype)

    monkeypatch.setattr(FormatSwitchLinear, "__call__", spy_call)
    monkeypatch.setattr(fused.down_proj, "_call_prepared", spy_prepared)

    x = mx.random.normal((1, 4, K)).astype(mx.bfloat16)
    inds = mx.random.randint(0, E, (1, 4, TOP_K)).astype(mx.uint32)
    scores = mx.softmax(mx.random.normal((1, 4, TOP_K)), axis=-1)
    out_gu = fused(x, inds)
    out_ws = qwen35_moe_weighted_sum._native_switch_weighted_sum(
        fused, x, inds, scores, lambda sorted_out, inv, s: sorted_out
    )
    assert out_gu.size == 0 and out_ws.size == 0
    assert len(declined) == 2
    assert direct_prepared == []
    assert len(via_call) == 2
