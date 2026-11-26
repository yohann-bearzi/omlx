# SPDX-License-Identifier: Apache-2.0
"""The versioned input transform: the descriptor, the memo and handoff, the
per-module binding, and the wrong-way guard a route that forgets the
rotation cannot pass.
"""

from __future__ import annotations

import gc

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx import formats
from omlx.formats import testing, transforms
from omlx.formats.linear import FormatLinear

_BLOCK = 128


class _Rotated(formats.QuantFormat):
    """A rotation's signs and a matrix already stored in the rotated basis."""

    mode = "toy_rotated"
    contract = formats.CONTRACT
    codes_name = "codes"
    tensors = ()
    shared_tensors = ("signs",)

    def layout(self, k, n, bits, group_size):
        if k % _BLOCK:
            return None
        return {
            self.codes_name: ((n, k), mx.float32),
            "signs": ((k // _BLOCK, _BLOCK), mx.float32),
        }

    def decode(self, tensors, bits, group_size):
        return tensors[self.codes_name]

    def input_transform(self, layer):
        return transforms.SignedWhtTransform(
            block=layer["signs"].shape[-1], tensors=("signs",)
        )


def _signs(k, seed=0):
    mx.random.seed(seed)
    return mx.array([1.0 if v > 0 else -1.0 for v in mx.random.normal((k,))]).reshape(
        -1, _BLOCK
    )


def _layer(fmt=None, k=256, n=8):
    fmt = fmt or _Rotated()
    w_rot = mx.random.normal((n, k)).astype(mx.float32) * 0.1
    tensors = {fmt.codes_name: w_rot, "signs": _signs(k)}
    with testing.verified("example"):
        return FormatLinear(fmt, 8, 1, k, tensors)


@pytest.fixture
def registered(monkeypatch):
    formats.load()
    monkeypatch.setattr(formats, "_FORMATS", dict(formats._FORMATS))
    fmt = _Rotated()
    formats.register(fmt)
    return fmt


def test_the_descriptor_is_versioned_and_reads_only_its_own_records():
    tx = transforms.SignedWhtTransform(block=256, tensors=("signs",))
    assert tx.version == transforms.TRANSFORM_VERSION == 1
    assert tx.record() == {"version": 1, "kind": "signed_wht", "block": 256}
    assert transforms.read_record(tx.record()) == tx.record()
    assert transforms.read_record({**tx.record(), "kind": "mine"}, kinds=None)
    with pytest.raises(ValueError, match="version"):
        transforms.read_record({**tx.record(), "version": 2})
    with pytest.raises(ValueError, match="version"):
        transforms.read_record({**tx.record(), "version": None})
    with pytest.raises(ValueError, match="kind"):
        transforms.read_record({**tx.record(), "kind": "other"})
    with pytest.raises(ValueError, match="kind"):
        transforms.read_record({**tx.record(), "kind": "two words"})
    with pytest.raises(ValueError, match="block"):
        transforms.read_record({**tx.record(), "block": 0})
    with pytest.raises(ValueError, match="block"):
        transforms.read_record({**tx.record(), "block": 512})
    with pytest.raises(ValueError, match="version"):
        transforms.InputTransform(version=2, block=128, apply=lambda x: x)
    with pytest.raises(ValueError, match="block"):
        transforms.InputTransform(block=0, apply=lambda x: x)


def test_the_memo_is_thread_local_and_drops_with_its_plain_array():
    transforms.clear()
    plain, rotated = mx.ones((2, 256)), mx.zeros((2, 256))
    key = ("k",)
    transforms.remember(plain, rotated, key)
    assert transforms.recall(plain, key) is rotated
    assert transforms.recall(mx.ones((2, 256)), key) is None
    assert transforms.recall(plain, ("other",)) is None
    del plain
    gc.collect()
    assert transforms.recall(mx.ones((2, 256)), key) is None
    transforms.clear()


def test_a_handoff_is_taken_once_and_shaped_like_its_reader():
    transforms.clear()
    key = ("k",)
    handed = mx.arange(24).reshape(4, 6)
    transforms.hand_off(key, handed)
    assert transforms.take(key, mx.zeros((24,), dtype=handed.dtype)) is not None
    # Taken: the next reader rotates itself.
    assert transforms.take(key, mx.zeros((24,), dtype=handed.dtype)) is None
    transforms.hand_off(key, handed)
    assert transforms.take(("other",), mx.zeros((24,), dtype=handed.dtype)) is None
    assert transforms.take(key, mx.zeros((7,), dtype=handed.dtype)) is None
    assert transforms.take(key, mx.zeros((24,), dtype=handed.dtype)).shape == (24,)
    transforms.clear()


def test_a_format_layer_binds_its_transform_once_and_plain_layers_have_none(
    registered, monkeypatch
):
    layer = _layer(registered)
    first = formats.input_transform(layer)
    second = formats.input_transform(layer)
    assert first is second
    tx, tensors = first
    assert isinstance(tx, transforms.InputTransform) and tx.block == _BLOCK
    assert tensors == (layer["signs"],)
    assert formats.input_transform(nn.Linear(8, 8)) is None
    quantized = nn.QuantizedLinear(32, 8, bias=False, group_size=32, bits=4)
    assert formats.input_transform(quantized) is None
    assert (
        formats.input_transform(nn.QuantizedLinear(64, 8, group_size=32, bits=4))
        is None
    )

    # A transform that reads a tensor its format does not share is refused.
    class _Stray(_Rotated):
        mode = "toy_rotated_stray"
        shared_tensors = ()

        def layout(self, k, n, bits, group_size):
            return {self.codes_name: ((n, k), mx.float32)}

        def input_transform(self, layer):
            return transforms.SignedWhtTransform(block=_BLOCK, tensors=("codes",))

    stray = _Stray()
    formats.register(stray)
    w_rot = mx.random.normal((8, 256)).astype(mx.float32)
    with testing.verified("example"):
        broken = FormatLinear(stray, 8, 1, 256, {stray.codes_name: w_rot})
    with pytest.raises(ValueError, match="shared_tensors"):
        formats.input_transform(broken)


class _Run:
    """A layer-shaped object whose ``__call__`` a test writes: the helper
    reads ``fmt``, the bounds' arrays, ``_decoded`` and the bias."""

    def __init__(self, layer, forgot):
        self.layer, self.forgot = layer, forgot
        self.fmt = layer.fmt
        self.bits, self.group_size = layer.bits, layer.group_size

    def __getitem__(self, name):
        return self.layer[name]

    def get(self, name, default=None):
        return self.layer.get(name, default)

    def _decoded(self):
        return self.layer._decoded()

    def __call__(self, x):
        tx, tensors = formats.input_transform(self)
        if not self.forgot:
            x = tx.apply(x, *tensors)
        return x @ self._decoded().T


def test_the_wrong_way_guard_fails_a_route_that_forgot_to_rotate(registered):
    layer = _layer(registered)
    x = mx.random.normal((1, 256)).astype(mx.float32) * 0.1
    # The plain-basis answer is the rotated storage read with rotated x.
    good = _Run(layer, forgot=False)
    testing.assert_transform_applied(good, x)
    bad = _Run(layer, forgot=True)
    with pytest.raises(AssertionError, match="not applied"):
        testing.assert_transform_applied(bad, x)


def test_prepare_applies_one_rotation_and_reuses_it(registered, monkeypatch):
    layer = _layer(registered)
    calls = []
    tx, tensors = formats.input_transform(layer)
    counting = transforms.InputTransform(
        version=tx.version,
        kind=tx.kind,
        tensors=tx.tensors,
        block=tx.block,
        apply=lambda x, *rest: calls.append(x) or tx.apply(x, *rest),
    )
    monkeypatch.setattr(formats, "input_transform", lambda layer: (counting, tensors))
    transforms.clear()
    x = mx.random.normal((2, 256)).astype(mx.float32)
    first = transforms.prepare(layer, x)
    second = transforms.prepare(layer, x)
    assert len(calls) == 1
    assert mx.array_equal(first, second).item()
    # A producer's handoff is the rotation, taken once.
    transforms.clear()
    other = mx.random.normal((2, 256)).astype(mx.float32)
    transforms.hand_off(transforms.key(counting, tensors), first)
    assert transforms.prepare(layer, other) is not None
    assert len(calls) == 1
    transforms.prepare(layer, other)
    assert len(calls) == 1


def test_same_checks_layer_tensors_when_descriptor_is_reused(registered):
    a = _layer(registered)
    b = _layer(registered)
    b["signs"] = -a["signs"]
    tx = transforms.SignedWhtTransform(block=_BLOCK, tensors=("signs",))
    assert tx.same(tx, a, a)
    assert not tx.same(tx, a, b)


def test_same_transform_handles_different_tensor_counts_and_value_equality(registered):
    class _NoTensor(_Rotated):
        mode = "toy_rotated_no_tensor"
        shared_tensors = ()

        def layout(self, k, n, bits, group_size):
            return {self.codes_name: ((n, k), mx.float32)}

        def input_transform(self, layer):
            return transforms.InputTransform(
                block=_BLOCK, tensors=(), apply=lambda x: x
            )

    no_tensor = _NoTensor()
    formats.register(no_tensor)
    a = _layer(registered)
    with testing.verified("example"):
        b = FormatLinear(
            no_tensor, 8, 1, 256, {no_tensor.codes_name: mx.ones((8, 256), mx.float32)}
        )
    assert transforms.same_transform(a, b) is False
    # Distinct arrays with identical sign values match via InputTransform.same.
    c = _layer(registered)
    c["signs"] = mx.array(a["signs"].tolist(), dtype=a["signs"].dtype)
    assert c["signs"] is not a["signs"]
    assert transforms.same_transform(a, c) is True


def test_check_transform_applied_reports_shape_mismatch(registered):
    layer = _layer(registered)

    class _WrongShape(_Run):
        def __call__(self, x):
            return mx.zeros((1, 16), dtype=x.dtype)

    with pytest.raises(AssertionError, match="shape"):
        testing.assert_transform_applied(
            _WrongShape(layer, forgot=False), mx.ones((1, 256))
        )


def test_version_refuses_bool_and_float():
    for bad in (True, 1.0):
        with pytest.raises(ValueError, match="version"):
            transforms.read_record({"version": bad, "kind": "signed_wht", "block": 128})
        with pytest.raises(ValueError, match="version"):
            transforms.InputTransform(version=bad, block=128, apply=lambda x: x)


def test_prepare_bound_with_key_skips_key_recompute(registered, monkeypatch):
    layer = _layer(registered)
    tx, tensors = formats.input_transform(layer)
    k = transforms.key(tx, tensors)
    monkeypatch.setattr(
        transforms,
        "key",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("recomputed")),
    )
    transforms.clear()
    x = mx.random.normal((1, 256)).astype(mx.float32)
    y = transforms.prepare_bound((tx, tensors, k), x)
    assert y.shape == x.shape


def test_take_checks_dtype_and_prepare_remembers_a_handoff():
    transforms.clear()
    key = ("k",)
    bf = mx.zeros((2, 256), mx.bfloat16)
    fp = mx.zeros((2, 256), mx.float32)
    transforms.hand_off(key, bf)
    assert transforms.take(key, fp) is None
    rotated = mx.ones((2, 256), mx.float32)
    transforms.hand_off(key, rotated)
    calls = []
    counter = transforms.InputTransform(
        block=128,
        apply=lambda x, *r: calls.append(x) or x,
    )
    bound = (counter, (), key)
    first = transforms.prepare_bound(bound, fp)
    second = transforms.prepare_bound(bound, fp)
    assert mx.array_equal(first, rotated).item()
    assert mx.array_equal(second, rotated).item()
    assert not calls
    transforms.clear()


def test_memo_pins_shared_tensors_while_plain_lives():
    import weakref

    transforms.clear()
    plain = mx.ones((2, 256), mx.float32)
    rotated = mx.zeros((2, 256), mx.float32)
    signs = _signs(256, seed=3)
    signs_ref = weakref.ref(signs)
    key = (1, "signed_wht", _BLOCK, (id(signs),))
    transforms.remember(plain, rotated, key, (signs,))
    del signs
    gc.collect()
    assert signs_ref() is not None
    assert transforms.recall(plain, key) is rotated
    transforms.clear()
    gc.collect()
    assert signs_ref() is None


def test_hand_off_pins_its_shared_tensors():
    import weakref

    transforms.clear()
    signs = mx.ones((2, 128), mx.bfloat16)
    tx = transforms.InputTransform(block=128, apply=lambda x, *r: x)
    rot = mx.zeros((4, 256), mx.bfloat16)
    ref = weakref.ref(signs)
    transforms.hand_off(transforms.key(tx, (signs,)), rot, (signs,))
    del signs
    gc.collect()
    assert ref() is not None
    transforms.clear()
    gc.collect()
    assert ref() is None


def test_same_is_symmetric_with_a_custom_same_as(registered):
    a = transforms.InputTransform(block=128, apply=lambda x, *r: x, same_as=None)
    b = transforms.InputTransform(
        block=128, apply=lambda x, *r: x, same_as=lambda *args: False
    )
    la = lb = object()
    assert a.same(b, la, lb) is False
    assert b.same(a, lb, la) is False


def test_memo_does_not_collide_for_distinct_custom_or_tensorless_transforms():
    transforms.clear()
    x = mx.ones((1, 256), mx.float32)
    plus = transforms.InputTransform(
        kind="custom", block=128, tensors=(), apply=lambda v: v
    )
    minus = transforms.InputTransform(
        kind="custom", block=128, tensors=(), apply=lambda v: -v
    )
    first = transforms.prepare_bound((plus, (), transforms.key(plus, ())), x)
    second = transforms.prepare_bound((minus, (), transforms.key(minus, ())), x)
    assert mx.array_equal(first, x).item()
    assert mx.array_equal(second, -x).item()

    signs = mx.ones((2, 128), mx.bfloat16)
    c_plus = transforms.InputTransform(
        kind="custom", block=128, tensors=("signs",), apply=lambda v, *r: v
    )
    c_minus = transforms.InputTransform(
        kind="custom", block=128, tensors=("signs",), apply=lambda v, *r: -v
    )
    transforms.clear()
    got_cp = transforms.prepare_bound(
        (c_plus, (signs,), transforms.key(c_plus, (signs,))), x
    )
    got_cm = transforms.prepare_bound(
        (c_minus, (signs,), transforms.key(c_minus, (signs,))), x
    )
    assert mx.array_equal(got_cp, x).item()
    assert mx.array_equal(got_cm, -x).item()

    a = transforms.InputTransform(
        block=128,
        tensors=("signs",),
        apply=lambda v, *r: v,
        same_as=lambda *args: False,
    )
    b = transforms.InputTransform(
        block=128,
        tensors=("signs",),
        apply=lambda v, *r: -v,
        same_as=lambda *args: False,
    )
    transforms.clear()
    got_a = transforms.prepare_bound((a, (signs,), transforms.key(a, (signs,))), x)
    got_b = transforms.prepare_bound((b, (signs,), transforms.key(b, (signs,))), x)
    assert mx.array_equal(got_a, x).item()
    assert mx.array_equal(got_b, -x).item()

    def shared_apply(v, *r):
        return v

    def shared_same_as(*args):
        return False

    eq_a = transforms.InputTransform(
        block=128,
        tensors=("signs",),
        apply=shared_apply,
        same_as=shared_same_as,
    )
    eq_b = transforms.InputTransform(
        block=128,
        tensors=("signs",),
        apply=shared_apply,
        same_as=shared_same_as,
    )
    assert eq_a == eq_b and eq_a is not eq_b
    assert transforms.key(eq_a, (signs,)) != transforms.key(eq_b, (signs,))


def test_same_requires_matching_shared_dtype_and_mutual_same_as(
    registered,
):
    class _Stub:
        def __init__(self, sign, dtype):
            self.value = mx.ones((2, 128), dtype) * sign

        def __getitem__(self, name):
            return self.value

    a = transforms.InputTransform(
        block=128,
        tensors=("signs",),
        apply=lambda v, *r: v,
        same_as=lambda *args: True,
    )
    b = transforms.InputTransform(
        block=128,
        tensors=("signs",),
        apply=lambda v, *r: v,
        same_as=lambda *args: False,
    )
    la, la2, lb = (
        _Stub(1, mx.bfloat16),
        _Stub(1, mx.bfloat16),
        _Stub(1, mx.float16),
    )
    assert a.same(b, la, la2) is False
    assert b.same(a, la2, la) is False
    assert (
        a.same(
            transforms.InputTransform(
                block=128, tensors=("signs",), apply=lambda v, *r: v
            ),
            la,
            lb,
        )
        is False
    )
    plain = transforms.InputTransform(
        block=128, tensors=("signs",), apply=lambda v, *r: v
    )
    assert plain.same(plain, la, lb) is False
    assert plain.same(plain, la, la2) is True


def test_same_distinguishes_custom_kind_and_tensorless_transforms():
    class _Stub:
        def __init__(self, signs):
            self.signs = signs

        def __getitem__(self, name):
            return self.signs

    signs = mx.ones((2, 128), mx.bfloat16)
    la, lb = _Stub(signs), _Stub(signs)
    a = transforms.InputTransform(
        kind="custom", block=128, tensors=(), apply=lambda v: v
    )
    b = transforms.InputTransform(
        kind="custom", block=128, tensors=(), apply=lambda v: -v
    )
    t1 = transforms.InputTransform(
        kind="signed_wht", block=128, tensors=(), apply=lambda v: v
    )
    t2 = transforms.InputTransform(
        kind="signed_wht", block=128, tensors=(), apply=lambda v: -v
    )
    c = transforms.InputTransform(
        kind="custom", block=128, tensors=("signs",), apply=lambda v, s: v
    )
    d = transforms.InputTransform(
        kind="custom", block=128, tensors=("signs",), apply=lambda v, s: -v
    )
    assert a.same(b, la, lb) is False
    assert t1.same(t2, la, lb) is False
    assert c.same(d, la, lb) is False
    assert a.same(a, la, lb) is True
    assert t1.same(t1, la, lb) is True
    assert c.same(c, la, lb) is True


def test_inverse_weight_missing_message_is_not_test_only():
    tx = transforms.InputTransform(block=128, apply=lambda x: x)
    with pytest.raises(ValueError, match="cannot recover") as exc:
        transforms.inverse_weight(mx.zeros((2, 128)), tx, ())
    assert "test" not in str(exc.value).lower()


def test_warm_model_warms_apply_split_swiglu_and_producers(monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("requires Metal")

    from mlx_vlm.models.switch_layers import SwiGLU

    from omlx.formats import kernels
    from omlx.formats.linear import FormatSwitchLinear
    from omlx.patches import qwen35_moe_routed_decode as routed

    counts: dict[str, list] = {
        "apply": [],
        "split_swiglu": [],
        "rmsnorm": [],
        "gated_norm": [],
        "act_pair": [],
    }

    def _apply(x, signs):
        counts["apply"].append((tuple(x.shape), x.dtype))
        return x * signs.reshape(-1).astype(x.dtype)

    def _split_swiglu(y, signs, *, limit=None):
        counts["split_swiglu"].append((tuple(y.shape), y.dtype, limit))
        return y[..., : signs.size]

    def _rmsnorm(x, weight, signs, *, eps):
        counts["rmsnorm"].append((tuple(x.shape), x.dtype, eps))
        return x, x

    def _gated_norm(x, gate, weight, signs, *, eps):
        counts["gated_norm"].append((tuple(x.shape), x.dtype, gate.dtype, eps))
        return gate, gate

    def _act_pair(a, b, signs, *, limit=None):
        counts["act_pair"].append((tuple(a.shape), a.dtype, limit))
        return a

    spec = transforms.ProducerSpec(
        rmsnorm=_rmsnorm,
        rmsnorm_block=128,
        rmsnorm_max_width=8192,
        gated_norm=_gated_norm,
        gated_norm_head_dim=128,
        act_pair=_act_pair,
    )
    counting_tx = transforms.InputTransform(
        block=128,
        tensors=("signs",),
        apply=_apply,
        apply_split_swiglu=_split_swiglu,
        producer_spec=lambda layer=None: spec,
    )

    class _CountingFmt(formats.QuantFormat):
        mode = "toy_counting_warm"
        contract = formats.CONTRACT
        codes_name = "codes"
        tensors = ()
        shared_tensors = ("signs",)

        def layout(self, k, n, bits, group_size):
            if k % 128:
                return None
            return {
                "codes": ((n, k), mx.bfloat16),
                "signs": ((k // 128, 128), mx.bfloat16),
            }

        def decode(self, tensors, bits, group_size):
            return tensors["codes"]

        def input_transform(self, layer):
            return counting_tx

    class _GatedNorm(nn.Module):
        def __init__(self, dim=128):
            super().__init__()
            self.weight = mx.ones((dim,), mx.bfloat16)
            self.eps = 1e-6

        def __call__(self, x, gate=None):
            return x

    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP

    def _make_qwen_mlp(fmt, signs_256, signs_128):
        mlp = Qwen3_5MLP(256, 128)
        with testing.verified("example"):
            mlp.gate_proj = FormatLinear(
                fmt,
                8,
                1,
                256,
                {"codes": mx.zeros((128, 256), mx.bfloat16), "signs": signs_256},
            )
            mlp.up_proj = FormatLinear(
                fmt,
                8,
                1,
                256,
                {"codes": mx.zeros((128, 256), mx.bfloat16), "signs": signs_256},
            )
            mlp.down_proj = FormatLinear(
                fmt,
                8,
                1,
                128,
                {"codes": mx.zeros((256, 128), mx.bfloat16), "signs": signs_128},
            )
        return mlp

    class _SwitchMLP(nn.Module):
        def __init__(self, fmt, signs_256, signs_128):
            super().__init__()
            self.activation = SwiGLU()
            with testing.verified("example"):
                self.gate_up_proj = FormatSwitchLinear(
                    fmt,
                    8,
                    1,
                    256,
                    {
                        "codes": mx.zeros((4, 256, 256), mx.bfloat16),
                        "signs": signs_256,
                    },
                )
                self.down_proj = FormatSwitchLinear(
                    fmt,
                    8,
                    1,
                    128,
                    {
                        "codes": mx.zeros((4, 256, 128), mx.bfloat16),
                        "signs": signs_128,
                    },
                )

    class _MiniModel(nn.Module):
        def __init__(self, fmt):
            super().__init__()
            signs_256 = mx.ones((2, 128), mx.bfloat16)
            signs_128 = mx.ones((1, 128), mx.bfloat16)
            self.attn_layer = nn.Module()
            self.attn_layer.input_layernorm = nn.RMSNorm(256)
            self.attn_layer.input_layernorm.weight = mx.ones((256,), mx.bfloat16)
            self.attn_layer.self_attn = nn.Module()
            with testing.verified("example"):
                self.attn_layer.self_attn.q_proj = FormatLinear(
                    fmt,
                    8,
                    1,
                    256,
                    {"codes": mx.zeros((64, 256), mx.bfloat16), "signs": signs_256},
                )
                self.attn_layer.self_attn.k_proj = FormatLinear(
                    fmt,
                    8,
                    1,
                    256,
                    {"codes": mx.zeros((64, 256), mx.bfloat16), "signs": signs_256},
                )
            self.gdn_layer = nn.Module()
            self.gdn_layer.linear_attn = nn.Module()
            self.gdn_layer.linear_attn.norm = _GatedNorm(128)
            with testing.verified("example"):
                self.gdn_layer.linear_attn.out_proj = FormatLinear(
                    fmt,
                    8,
                    1,
                    256,
                    {"codes": mx.zeros((64, 256), mx.bfloat16), "signs": signs_256},
                )
            self.mlp = _make_qwen_mlp(fmt, signs_256, signs_128)
            self.switch_holder = nn.Module()
            self.switch_holder.switch_mlp = _SwitchMLP(fmt, signs_256, signs_128)

    fmt = _CountingFmt()
    model = _MiniModel(fmt)
    installed = transforms.install_producers(model)
    assert installed == 3
    assert isinstance(model.attn_layer.input_layernorm, transforms._DualNorm)
    assert isinstance(model.gdn_layer.linear_attn.norm, transforms._DualGatedNorm)
    assert isinstance(model.mlp, transforms._FusedMLP)

    # Submitted closures must never capture live mx.array tensors.
    orig_submit = kernels._WARM_POOL.submit
    captured_arrays: list = []

    def _inspect(obj, seen):
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        if isinstance(obj, mx.array):
            captured_arrays.append(obj)
        elif isinstance(obj, (tuple, list, set, frozenset)):
            for item in obj:
                _inspect(item, seen)
        elif isinstance(obj, dict):
            for val in obj.values():
                _inspect(val, seen)
        elif callable(obj):
            for cell in getattr(obj, "__closure__", None) or ():
                _inspect(cell.cell_contents, seen)
            for default in getattr(obj, "__defaults__", None) or ():
                _inspect(default, seen)

    def _spy_submit(fn, *args, **kwargs):
        seen: set[int] = set()
        _inspect(fn, seen)
        _inspect(args, seen)
        _inspect(kwargs, seen)
        return orig_submit(fn, *args, **kwargs)

    monkeypatch.setattr(kernels._WARM_POOL, "submit", _spy_submit)

    routed.warm_moe_model(model, warm_dispatch=False)

    assert not captured_arrays, "warm closure captured live mx.array"
    # Deduplicated: only 2 distinct apply geometries (k=256 and k=128), both bf16.
    assert sorted(counts["apply"]) == [
        ((1, 128), mx.bfloat16),
        ((1, 256), mx.bfloat16),
    ]
    assert counts["split_swiglu"] == [((1, 256), mx.bfloat16, None)]
    assert counts["rmsnorm"] == [
        ((1, 256), mx.bfloat16, model.attn_layer.input_layernorm.eps)
    ]
    assert sorted(counts["gated_norm"], key=lambda item: str(item[1])) == sorted(
        [
            (
                (1, 1, 2, 128),
                mx.bfloat16,
                mx.bfloat16,
                model.gdn_layer.linear_attn.norm.eps,
            ),
            (
                (1, 1, 2, 128),
                mx.float32,
                mx.bfloat16,
                model.gdn_layer.linear_attn.norm.eps,
            ),
        ],
        key=lambda item: str(item[1]),
    )
    assert counts["act_pair"] == [((1, 128), mx.bfloat16, None)]

    # Fail closed: a transform warm error aborts warm_moe_model.
    broken_tx = transforms.InputTransform(
        block=128,
        tensors=("signs",),
        apply=lambda x, s: (_ for _ in ()).throw(RuntimeError("tx warm boom")),
    )

    class _BrokenFmt(_CountingFmt):
        mode = "toy_broken_warm"

        def input_transform(self, layer):
            return broken_tx

    class _BrokenModel(nn.Module):
        def __init__(self):
            super().__init__()
            with testing.verified("example"):
                self.proj = FormatLinear(
                    _BrokenFmt(),
                    8,
                    1,
                    128,
                    {
                        "codes": mx.zeros((64, 128), mx.bfloat16),
                        "signs": mx.ones((1, 128), mx.bfloat16),
                    },
                )

    with pytest.raises(RuntimeError, match="tx warm boom"):
        routed.warm_moe_model(_BrokenModel(), warm_dispatch=False)


def test_warm_model_runs_real_signed_wht_native_kernels():
    if not mx.metal.is_available():
        pytest.skip("requires Metal")

    from mlx_vlm.models.switch_layers import SwiGLU

    from omlx.formats import oqh
    from omlx.formats.linear import FormatSwitchLinear
    from omlx.patches import qwen35_moe_routed_decode as routed
    from tests.test_formats_oqh import _group_case

    def _oqh_lin(k, n, block, seed, signs=None):
        tensors, _ = _group_case(k, n, 4, 128, seed=seed, block=block)
        if signs is not None:
            tensors["oqh_signs"] = signs
        reader = oqh.OqhAffineFormat()
        reader.block = block
        with testing.verified("example"):
            return FormatLinear(reader, 4, 128, k, tensors)

    def _oqh_sw(k, n, experts, block, seed):
        parts = [
            _group_case(k, n, 4, 128, seed=seed + e, block=block)[0]
            for e in range(experts)
        ]
        tensors = {
            "weight": mx.stack([p["weight"] for p in parts]),
            "oqh_scales": mx.stack([p["oqh_scales"] for p in parts]),
            "oqh_biases": mx.stack([p["oqh_biases"] for p in parts]),
            "oqh_signs": parts[0]["oqh_signs"],
        }
        reader = oqh.OqhAffineFormat()
        reader.block = block
        with testing.verified("example"):
            return FormatSwitchLinear(reader, 4, 128, k, tensors)

    class _GatedNorm(nn.Module):
        def __init__(self, dim=128):
            super().__init__()
            self.weight = mx.ones((dim,), mx.bfloat16)
            self.eps = 1e-6

        def __call__(self, x, gate=None):
            return x

    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP

    def _make_real_mlp():
        mlp = Qwen3_5MLP(256, 128)
        mlp.gate_proj = _oqh_lin(256, 128, 256, seed=3)
        mlp.up_proj = _oqh_lin(256, 128, 256, seed=4, signs=mlp.gate_proj["oqh_signs"])
        mlp.down_proj = _oqh_lin(128, 256, 128, seed=5)
        return mlp

    class _RealOqhModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_layer = nn.Module()
            self.attn_layer.input_layernorm = nn.RMSNorm(256)
            self.attn_layer.input_layernorm.weight = mx.ones((256,), mx.bfloat16)
            self.attn_layer.self_attn = nn.Module()
            self.attn_layer.self_attn.q_proj = _oqh_lin(256, 64, 256, seed=1)
            self.gdn_layer = nn.Module()
            self.gdn_layer.linear_attn = nn.Module()
            self.gdn_layer.linear_attn.norm = _GatedNorm(128)
            self.gdn_layer.linear_attn.out_proj = _oqh_lin(256, 64, 128, seed=2)
            self.mlp = _make_real_mlp()
            self.switch_holder = nn.Module()
            self.switch_holder.switch_mlp = nn.Module()
            self.switch_holder.switch_mlp.activation = SwiGLU()
            self.switch_holder.switch_mlp.down_proj = _oqh_sw(128, 256, 4, 128, seed=10)

    model = _RealOqhModel()
    assert transforms.install_producers(model) == 3
    routed.warm_moe_model(model, warm_dispatch=False)


def test_plain_format_input_transform_short_circuits_cache():
    from omlx.formats import affine

    w_fp = mx.zeros((8, 64), mx.bfloat16)
    w, scales, biases = mx.quantize(w_fp, group_size=64, bits=4)
    fmt = affine.AffineFormat()
    with testing.verified("example"):
        layer = FormatLinear(
            fmt, 4, 64, 64, {"weight": w, "scales": scales, "biases": biases}
        )
    assert formats.input_transform(layer, refuse=False) is None
    assert "_omlx_input_transform" not in layer.__dict__

    # An instance-level input_transform override is never bypassed.
    called = []
    override_fmt = affine.AffineFormat()
    override_fmt.input_transform = lambda m: called.append(m) or None
    with testing.verified("example"):
        override_layer = FormatLinear(
            override_fmt,
            4,
            64,
            64,
            {"weight": w, "scales": scales, "biases": biases},
        )
    assert formats.input_transform(override_layer, refuse=False) is None
    assert called == [override_layer]


def test_same_passes_receiver_layer_first_to_other_same_as():
    class _Stub:
        def __init__(self, signs):
            self.signs = signs

        def __getitem__(self, name):
            return self.signs

    signs = mx.ones((2, 128), mx.bfloat16)
    la, lb = _Stub(signs), _Stub(signs)
    b_calls = []
    tx_a = transforms.InputTransform(
        block=128, tensors=("signs",), apply=lambda x, s: x, same_as=None
    )
    tx_b = transforms.InputTransform(
        block=128,
        tensors=("signs",),
        apply=lambda x, s: x,
        same_as=lambda other, first, second: (
            b_calls.append((other, first, second)) or (first is lb and second is la)
        ),
    )
    assert tx_a.same(tx_b, la, lb) is True
    assert b_calls == [(tx_a, lb, la)]

    a_calls, b_calls2 = [], []
    tx_a2 = transforms.InputTransform(
        block=128,
        tensors=("signs",),
        apply=lambda x, s: x,
        same_as=lambda other, first, second: (
            a_calls.append((other, first, second)) or (first is la and second is lb)
        ),
    )
    tx_b2 = transforms.InputTransform(
        block=128,
        tensors=("signs",),
        apply=lambda x, s: x,
        same_as=lambda other, first, second: (
            b_calls2.append((other, first, second)) or (first is lb and second is la)
        ),
    )
    assert tx_a2.same(tx_b2, la, lb) is True
    assert a_calls == [(tx_b2, la, lb)]
    assert b_calls2 == [(tx_a2, lb, la)]


def test_warm_model_skips_negative_and_invalid_switch_swiglu_limit(monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("requires Metal")

    from omlx.formats.linear import FormatSwitchLinear
    from omlx.patches import m5_gather_qmm

    seen_limits: list = []

    def _split_swiglu(y, signs, *, limit=None):
        seen_limits.append(limit)
        return y[..., : signs.size]

    tx = transforms.InputTransform(
        block=128,
        tensors=("signs",),
        apply=lambda x, s: x,
        apply_split_swiglu=_split_swiglu,
    )

    class _Fmt(formats.QuantFormat):
        mode = "toy_switch_limit_warm"
        contract = formats.CONTRACT
        codes_name = "codes"
        tensors = ()
        shared_tensors = ("signs",)

        def layout(self, k, n, bits, group_size):
            return {
                "codes": ((n, k), mx.bfloat16),
                "signs": ((k // 128, 128), mx.bfloat16),
            }

        def decode(self, tensors, bits, group_size):
            return tensors["codes"]

        def input_transform(self, layer):
            return tx

    class _SwitchHolder(nn.Module):
        def __init__(self, act):
            super().__init__()
            self.activation = act
            with testing.verified("example"):
                self.down_proj = FormatSwitchLinear(
                    _Fmt(),
                    8,
                    1,
                    128,
                    {
                        "codes": mx.zeros((4, 256, 128), mx.bfloat16),
                        "signs": mx.ones((1, 128), mx.bfloat16),
                    },
                )

    for bad in (-1.0, float("nan"), float("inf"), -float("inf"), True):
        seen_limits.clear()
        monkeypatch.setattr(m5_gather_qmm, "_swiglu_limit", lambda _act, b=bad: b)
        futs = transforms.warm_model(_SwitchHolder(object()), dtypes=(mx.bfloat16,))
        for f in futs:
            f.result()
        assert seen_limits == [], f"warmed invalid limit {bad!r}: {seen_limits}"

    for good in (0.0, 7.0):
        seen_limits.clear()
        monkeypatch.setattr(m5_gather_qmm, "_swiglu_limit", lambda _act, g=good: g)
        futs = transforms.warm_model(_SwitchHolder(object()), dtypes=(mx.bfloat16,))
        for f in futs:
            f.result()
        assert seen_limits == [good]


def test_dual_norm_inplace_weight_recast_recasts_tensors_atomically(monkeypatch):
    seen_dtypes = []
    cast_calls = []
    orig_cast = transforms._cast_tensors
    monkeypatch.setattr(
        transforms,
        "_cast_tensors",
        lambda tensors, dtype: cast_calls.append(dtype) or orig_cast(tensors, dtype),
    )

    def _stub_rmsnorm(x, weight, signs, *, eps):
        assert signs.dtype == weight.dtype
        assert x.dtype == weight.dtype
        seen_dtypes.append((weight.dtype, signs.dtype))
        return x, x

    spec = transforms.ProducerSpec(
        rmsnorm=_stub_rmsnorm, rmsnorm_block=128, rmsnorm_max_width=8192
    )
    stub_tx = transforms.InputTransform(
        block=128,
        tensors=("signs",),
        apply=lambda x, s: x,
        producer_spec=lambda layer=None: spec,
    )

    class _StubFmt(formats.QuantFormat):
        mode = "toy_dual_norm_cast"
        contract = formats.CONTRACT
        codes_name = "codes"
        tensors = ()
        shared_tensors = ("signs",)

        def layout(self, k, n, bits, group_size):
            return {
                "codes": ((n, k), mx.bfloat16),
                "signs": ((k // 128, 128), mx.bfloat16),
            }

        def decode(self, tensors, bits, group_size):
            return tensors["codes"]

        def input_transform(self, layer):
            return stub_tx

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.RMSNorm(256)
            self.input_layernorm.weight = mx.ones((256,), mx.bfloat16)
            self.self_attn = nn.Module()
            with testing.verified("example"):
                self.self_attn.q_proj = FormatLinear(
                    _StubFmt(),
                    8,
                    1,
                    256,
                    {
                        "codes": mx.zeros((64, 256), mx.bfloat16),
                        "signs": mx.ones((2, 128), mx.bfloat16),
                    },
                )

    blk = _Block()
    assert transforms.install_producers(blk) == 1
    norm = blk.input_layernorm
    assert isinstance(norm, transforms._DualNorm)
    assert hasattr(norm, "_omlx_tx_cast")
    assert not hasattr(norm, "_omlx_tx_weight_dtype")
    assert not hasattr(norm, "_omlx_tx_cast_tensors")
    assert cast_calls == [mx.bfloat16]

    transforms.clear()
    norm(mx.ones((1, 256), mx.bfloat16))
    norm(mx.ones((1, 256), mx.bfloat16))
    # Cache hits must not recast tensors.
    assert cast_calls == [mx.bfloat16]
    norm["weight"] = norm["weight"].astype(mx.float16)
    norm(mx.ones((1, 256), mx.float16))
    norm(mx.ones((1, 256), mx.float16))
    assert cast_calls == [mx.bfloat16, mx.float16]
    assert seen_dtypes == [
        (mx.bfloat16, mx.bfloat16),
        (mx.bfloat16, mx.bfloat16),
        (mx.float16, mx.float16),
        (mx.float16, mx.float16),
    ]
    assert norm._omlx_tx_cast[0] == mx.float16
    transforms.clear()
