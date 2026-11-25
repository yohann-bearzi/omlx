# SPDX-License-Identifier: Apache-2.0
"""Format layers rotate their input before any face reads it: every dense
and switch route produces the plain-basis product of the rotated weight,
the native fast layer is never registered over an unrotated call, and
direct-face patches (row_exact_qmv, gdn_prework, hc_fused) decline
transformed layers cleanly.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx import formats
from omlx.formats import linear as linear_mod
from omlx.formats import oqh, testing, transforms
from omlx.formats.linear import FormatEmbedding, FormatLinear, FormatSwitchLinear
from tests.test_formats_oqh import _group_case

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


def _rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.abs(a - b).max() / mx.abs(b).max()).item()


def _rotated(x, tensors):
    block = tensors["oqh_signs"].shape[-1]
    return transforms.SignedWhtTransform(block=block).apply(x, tensors["oqh_signs"])


def _dense(k, n, bits=4, seed=0):
    tensors, _ = _group_case(k, n, bits, 128, seed=seed)
    with testing.verified("example"):
        layer = FormatLinear(oqh.OQH_AFFINE, bits, 128, k, tensors)
    return layer, tensors


def _want_dense(tensors, x, bits=4):
    weight = oqh.OQH_AFFINE.decode(tensors, bits, 128)
    return _rotated(x, tensors) @ weight.astype(x.dtype).swapaxes(-1, -2)


@pytest.fixture(autouse=True)
def by_hand(monkeypatch):
    # A transformed layer never takes the native fast path: count its use.
    registered = []
    orig_dense = FormatLinear._register_fast
    orig_dot = FormatLinear._register_dot_fast

    def counted_dense(self, kept, plan, signature):
        registered.append(("dense", self, plan, signature))
        return orig_dense(self, kept, plan, signature)

    def counted_dot(self, kept, plan, signature):
        registered.append(("dot", self, plan, signature))
        return orig_dot(self, kept, plan, signature)

    monkeypatch.setattr(FormatLinear, "_register_fast", counted_dense)
    monkeypatch.setattr(FormatLinear, "_register_dot_fast", counted_dot)
    return registered


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8, 16])
def test_every_short_row_count_rotates_once(rows):
    layer, tensors = _dense(2560, 64)
    x = mx.random.normal((rows, 2560)).astype(mx.bfloat16)
    got = layer(x)
    assert _rel(got, _want_dense(tensors, x)) < 3e-2
    testing.assert_transform_applied(
        layer, mx.random.normal((1, 2560)).astype(mx.bfloat16), atol=5e-2
    )


def test_the_fast_layer_is_not_registered_over_a_transformed_input(by_hand):
    layer, tensors = _dense(2560, 64)
    # Exercise prefill (rows=32) and multi-row (rows=2,4,5,7,8), all of which
    # call _register_fast / _register_dot_fast on plain layers.
    x32 = mx.random.normal((32, 2560)).astype(mx.bfloat16)
    got1 = layer(x32)
    assert layer._fast is None and layer._fast2 is None
    got2 = layer(x32)
    assert _rel(got1, _want_dense(tensors, x32)) < 3e-2
    assert _rel(got2, _want_dense(tensors, x32)) < 3e-2
    for rows in (2, 4, 5, 7, 8):
        x = mx.random.normal((1, rows, 2560)).astype(mx.bfloat16)
        y1 = layer(x)
        assert layer._fast is None and layer._fast2 is None
        y2 = layer(x)
        assert _rel(y1, _want_dense(tensors, x)) < 3e-2
        assert _rel(y2, _want_dense(tensors, x)) < 3e-2
    assert by_hand, "expected multi-row/prefill to reach _register_*fast"
    assert layer._fast is None and layer._fast2 is None


def test_a_multi_axis_call_rotates_each_row_and_the_bias_rides_after():
    k, n = 640, 64
    tensors, _ = _group_case(k, n, 4, 128)
    layer, _ = _dense(k, n)
    layer["bias"] = mx.random.normal((n,)).astype(mx.bfloat16)
    x = mx.random.normal((2, 3, k)).astype(mx.bfloat16)
    got = layer(x)
    assert _rel(got, _want_dense(tensors, x) + layer["bias"]) < 3e-2


def test_qwen_routes_multi_row_and_prefill_rotate_once_without_mlx_bypass():
    layer, tensors = _dense(2560, 64)
    kept = layer._keep()
    assert kept.mlx_rows == frozenset()
    for shape in ((1, 5, 2560), (20, 2560)):
        x = mx.random.normal(shape).astype(mx.bfloat16)
        got = layer(x)
        want = _want_dense(tensors, x)
        assert _rel(got, want) < 3e-2


def test_dense_call_prepared_skips_double_rotation():
    layer, tensors = _dense(640, 64)
    x = mx.random.normal((2, 640)).astype(mx.bfloat16)
    rx = _rotated(x, tensors)
    got = layer._call_prepared(rx)
    assert _rel(got, _want_dense(tensors, x)) < 3e-2


def test_the_transform_is_looked_up_once_per_layer(monkeypatch):
    layer, _ = _dense(2560, 64)
    asked = []
    original = formats.input_transform
    monkeypatch.setattr(
        formats,
        "input_transform",
        lambda mod, **kw: asked.append(mod) or original(mod, **kw),
    )
    for rows in (1, 1, 1):
        mx.eval(layer(mx.random.normal((rows, 2560)).astype(mx.bfloat16)))
    assert asked.count(layer) == 1


def _switch(k, n, experts, bits=4):
    parts = [_group_case(k, n, bits, 128, seed=e) for e in range(experts)]
    tensors = {
        "weight": mx.stack([p[0]["weight"] for p in parts]),
        "oqh_scales": mx.stack([p[0]["oqh_scales"] for p in parts]),
        "oqh_biases": mx.stack([p[0]["oqh_biases"] for p in parts]),
        "oqh_signs": parts[0][0]["oqh_signs"],
    }
    with testing.verified("example"):
        layer = FormatSwitchLinear(oqh.OQH_AFFINE, bits, 128, k, tensors)
    return layer, tensors


def _want_switch(tensors, x, indices):
    weight = oqh.OQH_AFFINE.decode(tensors, 4, 128)
    xr = _rotated(x, tensors)
    return mx.gather_mm(
        xr, weight.astype(x.dtype).swapaxes(-1, -2), rhs_indices=indices
    )


def test_a_switch_decode_rotates_once_and_keeps_the_one_token_bits():
    layer, tensors = _switch(640, 64, 8)
    x = mx.random.normal((1, 1, 640)).astype(mx.bfloat16)
    indices = mx.array([0])
    got = layer(x, indices)
    assert _rel(got, _want_switch(tensors, x, indices)) < 3e-2
    # The same row alone in another batch is the same bits.
    assert mx.array_equal(got, layer(x, mx.array([0]))).item()


def test_a_switch_prefill_rotates_the_token_rows_once():
    layer, tensors = _switch(640, 64, 8)
    rows = 64
    x = mx.random.normal((rows, 1, 640)).astype(mx.bfloat16)
    indices = (mx.arange(rows, dtype=mx.uint32) * 3) % 8
    got = layer(x, indices, sorted_indices=False)
    weight = oqh.OQH_AFFINE.decode(tensors, 4, 128).astype(x.dtype)
    xr = _rotated(x, tensors)
    want = mx.stack([xr[i].reshape(1, 640) @ weight[indices[i]].T for i in range(rows)])
    assert _rel(got, want) < 3e-2


def test_a_prepared_call_does_not_rotate_again():
    layer, tensors = _switch(640, 64, 8)
    x = mx.random.normal((4, 1, 640)).astype(mx.bfloat16)
    indices = mx.array([[0], [1], [2], [3]])
    x_rot = _rotated(x, tensors)
    got = layer._call_prepared(x_rot, indices, sorted_indices=False)
    assert _rel(got, _want_switch(tensors, x, indices)) < 3e-2
    # A prepared call that is handed the plain input is the rotated product:
    # the caller owns the basis, so no second rotation hides a mistake.
    plain = layer._call_prepared(x, indices, sorted_indices=False)
    assert _rel(plain, _want_switch(tensors, x, indices)) > 1e-3


def test_a_switch_sorted_prefill_rotates_once():
    layer, tensors = _switch(640, 64, 8)
    rows = 64
    x = mx.random.normal((rows, 1, 640)).astype(mx.bfloat16)
    indices = mx.repeat(mx.arange(8, dtype=mx.uint32), rows // 8)
    got = layer(x, indices, sorted_indices=True)
    weight = oqh.OQH_AFFINE.decode(tensors, 4, 128).astype(x.dtype)
    xr = _rotated(x, tensors)
    want = mx.stack([xr[i].reshape(1, 640) @ weight[indices[i]].T for i in range(rows)])
    assert _rel(got, want) < 3e-2


def test_keep_no_poison_on_tx_error(monkeypatch):
    layer, tensors = _dense(640, 64)
    original = formats.input_transform
    state = {"failed": False}

    def fail_once(mod, **kw):
        if mod is layer and not state["failed"]:
            state["failed"] = True
            raise RuntimeError("transient tx error")
        return original(mod, **kw)

    monkeypatch.setattr(formats, "input_transform", fail_once)
    x = mx.random.normal((2, 640)).astype(mx.bfloat16)
    with pytest.raises(RuntimeError, match="transient tx error"):
        layer(x)
    assert "_omlx_format_call" not in layer.__dict__
    got = layer(x)
    assert _rel(got, _want_dense(tensors, x)) < 3e-2


def test_sorted_swiglu_rotates_fallback_x(monkeypatch):
    layer, tensors = _switch(640, 128, 4)
    rows = 32
    x = mx.random.normal((rows, 1, 640)).astype(mx.bfloat16)
    tokens = mx.random.normal((8, 1, 640)).astype(mx.bfloat16)
    row_map = mx.arange(rows, dtype=mx.uint32) % 8
    indices = mx.repeat(mx.arange(4, dtype=mx.uint32), rows // 4)
    recorded = []

    def fake_swiglu(plan, tile, arg_x, idx, limit, rm=None):
        if rm is not None:
            return None
        recorded.append(arg_x)
        return mx.zeros((rows, 1, 64), dtype=arg_x.dtype)

    monkeypatch.setattr(linear_mod, "_nax_host", lambda: True)
    monkeypatch.setattr(linear_mod, "_swiglu", fake_swiglu)
    out = layer._sorted_swiglu(x, indices, None, (tokens, row_map), prepared=False)
    assert out is not None and len(recorded) == 1
    assert mx.allclose(
        recorded[0].astype(mx.float32),
        _rotated(x, tensors).astype(mx.float32),
        atol=1e-3,
        rtol=1e-3,
    ).item()


def test_dense_decoded_rotates_after_load():
    formats.load()
    layer, tensors = _dense(640, 64)
    x = mx.random.normal((2, 640)).astype(mx.float32)
    assert _rel(layer(x), _want_dense(tensors, x)) < 1e-4


def test_embedding_as_linear_rotates():
    tensors, _ = _group_case(640, 64, 4, 128)
    with testing.verified("example"):
        emb = FormatEmbedding(oqh.OQH_AFFINE, 4, 128, 640, tensors)
    x = mx.random.normal((2, 640)).astype(mx.bfloat16)
    assert _rel(emb.as_linear(x), _want_dense(tensors, x)) < 3e-2


def test_invalid_shape_and_empty_inputs(monkeypatch):
    dense, _ = _dense(640, 64)
    switch, _ = _switch(640, 64, 4)
    with pytest.raises(ValueError, match="x is"):
        dense(mx.array(1.0, dtype=mx.bfloat16))
    with pytest.raises(ValueError, match="x is"):
        switch(mx.array(1.0, dtype=mx.bfloat16), mx.array([0], dtype=mx.uint32))
    with pytest.raises(ValueError, match="indices are integers"):
        switch(mx.zeros((1, 1, 640), mx.bfloat16), mx.array([0.0], dtype=mx.float32))
    called = []
    monkeypatch.setattr(transforms, "prepare_bound", lambda *a, **kw: called.append(a))
    out_d = dense(mx.zeros((0, 640), mx.bfloat16))
    assert out_d.shape == (0, 64) and not called
    out_s = switch(mx.zeros((0, 1, 640), mx.bfloat16), mx.zeros((0,), mx.uint32))
    assert out_s.shape == (0, 1, 64) and not called


def test_row_exact_qmv_declines_a_transformed_layer():
    from omlx.patches import row_exact_qmv

    l1, t1 = _dense(640, 64, seed=0)
    l2, t2 = _dense(640, 64, seed=1)
    x = mx.random.normal((2, 640)).astype(mx.bfloat16)
    got = row_exact_qmv.quantized_linear(l1, x)
    assert _rel(got, _want_dense(t1, x)) < 3e-2
    y1, y2 = row_exact_qmv.quantized_linears((l1, l2), x)
    assert _rel(y1, _want_dense(t1, x)) < 3e-2
    assert _rel(y2, _want_dense(t2, x)) < 3e-2
    assert row_exact_qmv.quantized_linears_tiled((l1,), x, 2, 4) is None
    assert row_exact_qmv.format_rows_qmv(l1, mx.bfloat16, lambda r: (2, 1)) is None
    # The plain-basis answer is still the layer's own call.
    assert _rel(l1(x), _want_dense(t1, x)) < 3e-2


def test_row_exact_refuses_transformed_placeholders():
    from omlx.patches import row_exact_qmv

    tensors, _ = _group_case(640, 64, 4, 128)
    with testing.verified("example"):
        layer = FormatLinear(oqh.OQH_AFFINE, 4, 128, 640, tensors, placeholders=True)
    with pytest.raises(ValueError, match="placeholder"):
        row_exact_qmv.format_rows_qmv(layer, mx.bfloat16, lambda r: (2, 1))


def test_gdn_stack_format_inputs_declines_transformed_layers():
    from types import SimpleNamespace

    from omlx.patches import qwen35_gdn_prework

    projs = [_dense(640, 64, seed=i)[0] for i in range(4)]
    for p in projs[1:]:
        p["oqh_signs"] = projs[0]["oqh_signs"]
    stub = SimpleNamespace(
        in_proj_qkv=projs[0],
        in_proj_z=projs[1],
        in_proj_b=projs[2],
        in_proj_a=projs[3],
        conv1d=SimpleNamespace(weight=mx.zeros((1,), mx.bfloat16)),
    )
    assert qwen35_gdn_prework._stack_format_inputs(stub) is None


def test_embedding_lookup_inverts_the_input_transform():
    from omlx.formats import transforms as tx
    from omlx.formats.linear import FormatEmbedding

    tensors, _ = _group_case(640, 64, 4, 128)
    with testing.verified("example"):
        emb = FormatEmbedding(oqh.OQH_AFFINE, 4, 128, 640, tensors)
    ids = mx.array([0, 1])
    bound = formats.input_transform(emb)
    want = tx.inverse_weight(emb._decoded(ids), *bound).astype(mx.bfloat16)
    assert _rel(emb(ids), want) < 1e-4
    assert emb(mx.array([], dtype=mx.uint32)).shape == (0, 640)


def test_gdn_stack_declines_when_a_non_first_layer_has_a_transform(monkeypatch):
    from types import SimpleNamespace

    from omlx.patches import qwen35_gdn_prework

    layers = []
    for _ in range(4):
        tensors, _ = _group_case(256, 64, 4, 128)
        with testing.verified("example"):
            layers.append(FormatLinear(oqh.OQH_AFFINE, 4, 128, 256, tensors))
    for layer in layers:
        layer._fake_tx = None
    layers[1]._fake_tx = object()
    monkeypatch.setattr(
        formats,
        "input_transform",
        lambda layer, **kwargs: getattr(layer, "_fake_tx", None),
    )
    stub = SimpleNamespace(
        in_proj_qkv=layers[0],
        in_proj_z=layers[1],
        in_proj_b=layers[2],
        in_proj_a=layers[3],
    )
    assert qwen35_gdn_prework._stack_format_inputs(stub) is None


def test_hc_fused_declines_transformed_projection():
    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import hc_fused

    layer, _ = _dense(256, 64, seed=3)
    assert hc_fused._projection_face(layer, 256, 64, mx.bfloat16, False) is None
    assert hc_fused._projection_face(layer, 256, 64, mx.bfloat16, True) is None


def test_publish_fast_blocks_transformed_kept_and_accepts_sentinel():
    from omlx.formats import linear

    layer, _ = _dense(256, 64, seed=4)
    kept = layer._keep()
    assert kept.tx is not None
    fake_fast = object()
    layer._publish_fast(kept, fake_fast)
    assert layer._fast is None and layer._fast2 is None

    class _Proxy:
        def __init__(self, store):
            object.__setattr__(self, "_store", store)

        @property
        def __dict__(self):
            return self._store

    sentinel_kept, sentinel_fast = object(), object()
    store = {linear._KEPT: sentinel_kept}
    FormatLinear._publish_fast(_Proxy(store), sentinel_kept, sentinel_fast)
    assert store.get(linear._FAST) is sentinel_fast
    assert linear._FAST2 not in store
