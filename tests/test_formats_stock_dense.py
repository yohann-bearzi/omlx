# SPDX-License-Identifier: Apache-2.0
"""Ordinary Qwen dense projections compared with MLX's public affine operation."""

import gc
import json
import logging
import weakref
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from threading import Barrier

import mlx.core as mx
import numpy as np
import pytest

from omlx import formats
from omlx.formats import kernels, linear, testing, tile
from omlx.formats.affine import AffineFormat
from omlx.formats.linear import FormatLinear
from omlx.patches.format_load import loading
from omlx.patches.qwen35_verify_qmm import set_verify_qmm_armed

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture
def build_layer(tmp_path, monkeypatch):
    from omlx.patches import format_load

    def build(fmt, bits, group, k, tensors, family="qwen4_exp"):
        mode = fmt if isinstance(fmt, str) else fmt.mode
        config = {"model_type": family, "omlx_quant_formats": {mode: {}}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        with monkeypatch.context() as patch:
            if mode == "affine":
                patch.setattr(format_load, "PORTED", format_load.PORTED | {family})
            with testing.verified(family), loading(tmp_path):
                return FormatLinear(fmt, bits, group, k, tensors)

    return build


def _projection(build, n, k, bits, dtype, *, paired_halves=False):
    keys = mx.random.split(mx.random.key(161 + n + bits))
    width = k // 2 if paired_halves else k
    w = mx.random.normal((n, width), key=keys[0]).astype(dtype)
    if paired_halves:
        w = mx.concatenate([w, w], axis=1)
    packed = mx.quantize(w, group_size=64, bits=bits)
    names = ("weight", "scales", "biases")
    layer = build("affine", bits, 64, k, dict(zip(names, packed)))
    return layer, packed, keys[1]


def _stock(x, packed, bits):
    return mx.quantized_matmul(x, *packed, transpose=True, bits=bits, group_size=64)


def _compare(layer, packed, x, monkeypatch, *, expected=None):
    want = _stock(x, packed, layer.bits) if expected is None else expected
    mx.eval(want)
    calls = []

    def observe(function):
        def run(*args, **kwargs):
            calls.append(function.__name__)
            return function(*args, **kwargs)

        return run

    with monkeypatch.context() as patch:
        for name in ("quantized_matmul", "gather_qmm", "dequantize"):
            patch.setattr(mx, name, observe(getattr(mx, name)))
        patch.setattr(FormatLinear, "_decoded", observe(FormatLinear._decoded))
        got = layer(x)
        mx.eval(got)

    assert got.shape == want.shape and got.dtype == want.dtype
    np.testing.assert_array_equal(
        np.asarray(got.view(mx.uint16)), np.asarray(want.view(mx.uint16))
    )
    assert not calls, f"dense format layer used {calls}"
    return got


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 17])
def test_wide_rows_match_stock(rows, bits, dtype, build_layer, monkeypatch):
    k, n = 2560, 512
    layer, packed, key = _projection(build_layer, n, k, bits, dtype, paired_halves=True)
    # Equal weight halves with opposite inputs expose differences in the
    # reduction order even when low-precision outputs usually round alike.
    half = mx.random.normal((rows, k // 2), key=key).astype(dtype)
    x = mx.concatenate([half, -half], axis=1)
    _compare(layer, packed, x, monkeypatch)


class _AffinePlugin(AffineFormat):
    mode = "stock_layout_control"


def _equivalent_layer(mode, build_layer, n, k, family="qwen4_exp", *, dtype=mx.float16):
    from tests.test_formats_kernels import _toy
    from tests.test_formats_tile_prefill import _TILE, _tile_toy, _ToyFormat

    keys = mx.random.split(mx.random.key(354), num=2)
    half = mx.random.randint(0, 256, shape=(n, k // 2), key=keys[0]).astype(mx.uint8)
    codes = mx.concatenate([half, half], axis=1)
    scales = mx.ones((n, k // 64), dtype=dtype)
    packed = (codes.view(mx.uint32), scales, -128 * scales)
    if mode in ("stock", "affine_plugin"):
        layer = build_layer(
            "affine" if mode == "stock" else _AffinePlugin(),
            8,
            64,
            k,
            dict(zip(("weight", "scales", "biases"), packed)),
            family=family,
        )
    else:
        # A plugin stores unsigned bytes and a single scale per row. Its two
        # faces represent exactly the same weights as the public stock oracle.
        tile = _tile_toy("stock_wide", _TILE.replace("GROUP = 8", "GROUP = 64"))
        plugin = _ToyFormat(_toy("stock_wide"), None if mode == "dot_only" else tile)
        layer = build_layer(
            plugin,
            8,
            64,
            k,
            {"codes": codes, "group_scales": mx.ones((n,))},
            family=family,
        )
    return layer, packed, keys[1]


class _StockLayoutPlugin(AffineFormat):
    """The affine layout and faces under a mode of the plugin's own: the stock
    switch does not claim it, so its dense route runs the format's tile."""

    mode = "stock_layout_control"


def _affine_plugin(build, n, k, bits, dtype):
    keys = mx.random.split(mx.random.key(161 + n + bits))
    w = mx.random.normal((n, k), key=keys[0]).astype(dtype)
    packed = mx.quantize(w, group_size=64, bits=bits)
    layer = build(
        _StockLayoutPlugin(),
        bits,
        64,
        k,
        dict(zip(("weight", "scales", "biases"), packed)),
    )
    return layer, packed, keys[1]


@pytest.mark.parametrize("mode", ["stock", "plugin"])
def test_ordinary_and_row_exact_calls_keep_separate_arithmetic(mode, build_layer):
    k, n, rows = 2560, 160, 5
    layer, packed, key = _equivalent_layer(mode, build_layer, n, k)
    half = mx.random.normal((rows, k // 2), key=key).astype(mx.float16)
    x = mx.concatenate([half, -half], axis=1)
    wide = mx.quantized_matmul(x, *packed, transpose=True, bits=8, group_size=64)
    serial = mx.concatenate([layer(x[i : i + 1]) for i in range(rows)])
    wide_bits = np.asarray(wide.view(mx.uint16))
    serial_bits = np.asarray(serial.view(mx.uint16))
    assert np.any(wide_bits != serial_bits), "fixture must distinguish the traversals"
    for exact in (False, True, False):
        set_verify_qmm_armed(exact, row_exact=exact)
        try:
            got = np.asarray(layer(x).view(mx.uint16))
            np.testing.assert_array_equal(got, serial_bits if exact else wide_bits)
        finally:
            set_verify_qmm_armed(False)
    face = tile(layer, k, n)
    assert any(
        name[:2] == ("format_linear_wide", face) and proof.ran
        for name, proof in kernels._PROOFS.items()
    )


@pytest.mark.parametrize("mode", ["stock", "plugin"])
def test_warm_wide_replays_until_its_proof_fails(mode, build_layer, monkeypatch):
    from omlx.formats import linear

    k, n = 2560, 160
    layer, packed, key = _equivalent_layer(mode, build_layer, n, k)
    half = mx.random.normal((1, 2, k // 2), key=key).astype(mx.float16)
    x = mx.concatenate([half, -half], axis=-1)
    wide = _stock(x, packed, 8)
    serial = mx.stack([layer(row) for row in x[0]])[None]
    assert not mx.array_equal(wide.view(mx.uint16), serial.view(mx.uint16)).item()
    selected = []
    select = linear.traversal

    def record(*args, **kwargs):
        selected.append(args)
        return select(*args, **kwargs)

    monkeypatch.setattr(linear, "traversal", record)
    _compare(layer, packed, x, monkeypatch)
    assert selected
    selected.clear()
    _compare(layer, packed, x, monkeypatch)
    assert not selected, "a warm wide call selected its traversal again"

    face = tile(layer, k, n)
    (proof,) = (
        p
        for name, p in kernels._PROOFS.items()
        if name[:2] == ("format_linear_wide", face)
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(proof.fail).result()
    _compare(layer, packed, x, monkeypatch, expected=serial)
    assert selected, "a retired wide graph bypassed route selection"


@pytest.mark.parametrize("mode", ["stock", "plugin"])
def test_qwen_threads_and_another_family_can_share_a_face(mode, build_layer):
    k, n, rows = 2560, 160, 5
    qwen, packed, key = _equivalent_layer(mode, build_layer, n, k)
    other, _, _ = _equivalent_layer(mode, build_layer, n, k, family="llama")
    half = mx.random.normal((rows, k // 2), key=key).astype(mx.float16)
    x = mx.concatenate([half, -half], axis=1)
    serial = mx.concatenate([qwen(x[i : i + 1]) for i in range(rows)])
    wide = mx.quantized_matmul(x, *packed, transpose=True, bits=8, group_size=64)
    mx.eval(serial, wide, x, qwen.parameters(), other.parameters())
    serial, wide = (np.asarray(a.view(mx.uint16)) for a in (serial, wide))
    assert np.any(wide != serial)
    barrier = Barrier(2, timeout=20)

    def run(exact):
        set_verify_qmm_armed(exact, row_exact=exact)
        try:
            barrier.wait()
            results = (qwen(x).view(mx.uint16), other(x).view(mx.uint16))
            mx.eval(*results)
            return results
        finally:
            set_verify_qmm_armed(False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, (True, False)))
    for (qwen_result, other_result), expected in zip(results, (serial, wide)):
        np.testing.assert_array_equal(np.asarray(other_result), serial)
        np.testing.assert_array_equal(np.asarray(qwen_result), expected)


def test_a_warm_row_exact_graph_preserves_dot_only_fallback_for_both_families(
    build_layer, monkeypatch
):
    k, n, rows = 2560, 160, 17
    qwen, _, key = _equivalent_layer("dot_only", build_layer, n, k)
    other, _, _ = _equivalent_layer("dot_only", build_layer, n, k, family="llama")
    half = mx.random.normal((rows, k // 2), key=key).astype(mx.float16)
    x = mx.concatenate([half, -half], axis=1)
    serial = mx.concatenate([qwen(x[i : i + 1]) for i in range(rows)])
    names = formats.tensor_names(qwen)
    decoded = (qwen[names[0]].astype(mx.float32) - 128) * qwen[names[1]][:, None]
    composed = x @ decoded.astype(x.dtype).T
    serial, composed = (np.asarray(y.view(mx.uint16)) for y in (serial, composed))
    assert np.any(serial != composed)
    decode = FormatLinear._decoded
    calls = []

    def observe(layer, *args):
        calls.append(id(layer))
        return decode(layer, *args)

    monkeypatch.setattr(FormatLinear, "_decoded", observe)
    # On this thread all these calls have the same Dot, shape and dtype.
    for exact, layer, expected, decodes in (
        (False, qwen, composed, True),
        (True, qwen, serial, False),
        (True, other, composed, True),
        (False, qwen, composed, True),
    ):
        calls.clear()
        set_verify_qmm_armed(exact, row_exact=exact)
        try:
            got = np.asarray(layer(x).view(mx.uint16))
        finally:
            set_verify_qmm_armed(False)
        np.testing.assert_array_equal(got, expected)
        assert calls == ([id(layer)] if decodes else [])


def _gapped_rows(rows, k, key, *, matrices=2):
    half = mx.random.normal((2 * matrices, rows, k // 2), key=key).astype(mx.float16)
    return mx.concatenate([half, -half], axis=-1)[::2]


@pytest.mark.parametrize("mode", ["stock", "affine_plugin", "plugin"])
def test_gapped_single_row_matrices_fold_to_the_dense_route(
    mode, build_layer, monkeypatch
):
    k, n = 2560, 48
    layer, packed, key = _equivalent_layer(mode, build_layer, n, k)
    # MLX rounds each gapped singleton matrix on its own; the single folded
    # route (rule 33) keeps the dense rounding and reorders inside the bar,
    # so the same data must answer alike in either layout.
    gapped = _gapped_rows(1, k, key, matrices=3)
    dense = mx.contiguous(gapped)
    folded = _stock(dense, packed, 8)
    stock = _stock(gapped, packed, 8)
    assert not mx.array_equal(stock.view(mx.uint16), folded.view(mx.uint16)).item()
    got = _compare(layer, packed, dense, monkeypatch)
    _compare(layer, packed, gapped, monkeypatch, expected=got)
    _compare(layer, packed, dense, monkeypatch)


@pytest.mark.parametrize("mode", ["stock", "plugin"])
def test_batched_format_calls_fold_to_the_dense_route(mode, build_layer, monkeypatch):
    rows, k, n = 5, 2560, 48
    layer, packed, key = _equivalent_layer(mode, build_layer, n, k)
    gapped = _gapped_rows(rows, k, key)
    dense = mx.contiguous(gapped)
    got = _compare(layer, packed, dense, monkeypatch)
    _compare(layer, packed, gapped, monkeypatch, expected=got)
    _compare(layer, packed, dense, monkeypatch)


@pytest.mark.parametrize("mode", ["stock", "plugin"])
def test_native_and_python_fallback_wide_dispatch_match_bitwise(
    mode, build_layer, monkeypatch, caplog
):
    from omlx.custom_kernels.quant_matmul import fast as native
    from omlx.formats import linear

    k, n = 2560, 160
    layer, packed, key = _equivalent_layer(mode, build_layer, n, k)
    half = mx.random.normal((2, k // 2), key=key).astype(mx.float16)
    x = mx.concatenate([half, -half], axis=-1)
    cold = _compare(layer, packed, x, monkeypatch)
    if native.has_symbol("FormatLinearFast"):
        assert layer._fast is not None

        def missed_native():
            pytest.fail("native warm call fell through to the Python lookup")

        object.__setattr__(layer, "_qwen_row_exact", missed_native)
    warm = _compare(layer, packed, x, monkeypatch, expected=cold)

    monkeypatch.setattr(native, "has_symbol", lambda name: False)
    monkeypatch.setattr(linear, "_NATIVE_FALLBACK_LOGGED", False)
    caplog.set_level("WARNING", logger=linear.__name__)
    caplog.clear()
    for _ in range(2):
        fallback, _, _ = _equivalent_layer(mode, build_layer, n, k)
        for _ in range(2):
            _compare(fallback, packed, x, monkeypatch, expected=warm)
        assert fallback._fast is None
    warnings = [
        r for r in caplog.records if "FormatLinearFast unavailable" in r.message
    ]
    assert len(warnings) == 1


def test_native_dispatch_releases_worker_graph_on_thread_exit(build_layer, monkeypatch):
    from omlx.custom_kernels.quant_matmul import fast as native

    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires the native warm dispatcher")
    k, n = 2560, 160
    layer, _, key = _equivalent_layer("stock", build_layer, n, k)
    layer.bias = mx.zeros((n,), dtype=mx.float16)
    x = mx.random.normal((2, k), key=key).astype(mx.float16)
    mx.eval(x, layer.parameters())
    register = FormatLinear._register_fast
    refs = []

    def observe(self, kept, tile, signature):
        graph = tile.proof.graphs[signature]

        def replay(*args):
            return graph(*args)

        refs.append(weakref.ref(replay))
        tile.proof.keep(signature, replay)
        register(self, kept, tile, signature)

    monkeypatch.setattr(FormatLinear, "_register_fast", observe)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(lambda: mx.eval(layer(x))).result()

        # Replay once on the installer thread so its native memo holds a hit.
        def replay():
            y = layer._fast(layer, x)
            assert y is not None
            mx.eval(y)

        pool.submit(replay).result()
    gc.collect()
    assert layer._fast is not None and len(refs) == 1
    assert refs[0]() is None, "native dispatcher retained the exited thread's graph"


def test_format_linear_fast_returns_none_on_a_four_arg_extension(monkeypatch, caplog):
    """An extension built before D11 has no packed-key argument: the wrapper
    returns None and the layer keeps its Python lookup."""
    from omlx.custom_kernels.quant_matmul import fast as native

    class Old:
        def __init__(self, weight, bias, operands, proof_state):
            self.args = (weight, bias, operands, proof_state)

        def bind(self):
            return lambda *args: None

    monkeypatch.setattr(native, "_FORMAT_LINEAR_FAST", Old)
    monkeypatch.setattr(native, "_STALE_PACKED_KEY_WARNED", False)
    weight = mx.zeros((8, 8), mx.uint8)
    with caplog.at_level(logging.WARNING, logger=native.__name__):
        for _ in range(2):
            assert (
                native.format_linear_fast(weight, None, [weight], {}, "codes") is None
            )
    # One warning, naming the stale build and how to rebuild it.
    (warning,) = caplog.records
    said = warning.getMessage()
    assert "make dev" in said and "packed-key" in said


def test_the_native_binding_is_bound_with_the_layers_packed_key():
    """D11: the replay reads the layer's dict by the format's codes name, so
    the product's wrapper requires it (no default), while the binding keeps
    its own default for a caller that predates D11."""
    from omlx.custom_kernels.quant_matmul import fast as native

    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires the native warm dispatcher")
    weight = mx.zeros((8, 8), mx.uint8)
    with pytest.raises(TypeError):
        native.format_linear_fast(weight, None, [weight], {})
    with pytest.raises(TypeError, match=r"weight_key must be str, not int"):
        native.format_linear_fast(weight, None, [weight], {}, 123)
    with pytest.raises(TypeError, match=r"weight_key must be str, not float"):
        native.format_linear_fast(weight, None, [weight], {}, 1.5)
    with pytest.raises(TypeError):
        native.format_linear_fast(weight, None, [weight], "not_a_dict", "codes")
    # A layer that keeps its codes under another name replays natively.
    dispatch = native.format_linear_fast(weight, None, [weight], {}, "codes")
    assert dispatch.__self__ is not None
    signature = ((1, 8), mx.float16)
    expected = mx.zeros((1, 8), mx.float16)
    proof = kernels.Proof()
    proof.keep(signature, lambda *args, result=expected: result)
    legacy = native._FORMAT_LINEAR_FAST(weight, None, [weight], vars(proof))
    replay = legacy.bind()
    legacy.register(signature)
    # Its own default must be "weight": the layer keeps its codes there.
    assert replay({"weight": weight}, mx.zeros((1, 8), mx.float16)) is expected


def test_native_last_wide_replay_misses_on_shape_or_dtype_change():
    from omlx.custom_kernels.quant_matmul import fast as native

    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires the native warm dispatcher")
    # Observe reads of the proof's actual worker-local state. Distinct array
    # objects with the same signature should replay without consulting it.
    reads = []

    class CountReads(kernels._Kept):
        def __getattribute__(self, name):
            reads.append(name)
            return super().__getattribute__(name)

    proof = kernels.Proof()
    proof._kept = CountReads()
    weight = object()
    layer = {"weight": weight}
    replay = native.format_linear_fast(weight, None, [weight], vars(proof), "weight")
    signatures = [
        ((m, 4), dtype)
        for m, dtype in [(2, mx.float16), (3, mx.float16), (3, mx.float32)]
    ]
    outputs = [object() for _ in signatures]
    for key, output in zip(signatures, outputs):
        proof.keep(key, lambda *args, result=output: result)
        replay.__self__.register(key)
    for i in (0, 1, 2, 0):
        reads.clear()
        assert replay(layer, mx.zeros(*signatures[i])) is outputs[i]
        assert reads, "changed signature reused the previous graph"
        reads.clear()
        assert replay(layer, mx.zeros(*signatures[i])) is outputs[i]
        assert not reads, "repeated signature looked up the proof again"


@pytest.mark.parametrize("mode", ["stock", "plugin"])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_two_request_decode_folds_one_graph_for_live_layouts(
    mode, dtype, build_layer, monkeypatch
):
    from omlx.custom_kernels.quant_matmul import fast as native

    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires the native warm dispatcher")
    k, n = 2560, 256
    layer, packed, key = _equivalent_layer(mode, build_layer, n, k, dtype=dtype)
    if mode == "stock":
        assert layer.scales is packed[1] and layer.biases is packed[2]
    # MLX rounds each gapped singleton matrix on its own; the fold answers
    # every layout with the dense rounding (rule 33). The folded 2-D call
    # registers the native warm dispatcher, so later calls replay it.
    gapped = mx.repeat(_gapped_rows(1, k, key).astype(dtype), 2, axis=0)[::2]
    dense = mx.contiguous(gapped)
    folded = _stock(dense, packed, 8)
    _compare(layer, packed, dense, monkeypatch, expected=folded)
    assert layer._fast is not None, "folded call registered no native replay"

    broadcast = mx.broadcast_to(gapped[:1], gapped.shape)
    column = mx.repeat(dense, 2, axis=-1)[..., ::2]
    folded_broadcast = _stock(
        mx.contiguous(broadcast.reshape(2, k)), packed, 8
    ).reshape(2, 1, n)
    layouts = [(dense, folded), (gapped, folded), (broadcast, folded_broadcast)]
    layouts.append((column, folded))
    for x, expected in layouts:
        got = layer._fast(layer, x)
        assert got is not None, "two-request layout missed native replay"
        np.testing.assert_array_equal(got.view(mx.uint16), expected.view(mx.uint16))
    traced = mx.compile(lambda value: layer(value))
    for x, expected in layouts:
        np.testing.assert_array_equal(
            traced(x).view(mx.uint16), expected.view(mx.uint16)
        )

    ordinary = dense.reshape(2, k)
    for x in (ordinary, ordinary):
        _compare(layer, packed, x, monkeypatch, expected=folded[:, 0])
        native_y = layer._fast(layer, x)
        assert native_y is not None, "warm call kept a different plan's dispatcher"
        np.testing.assert_array_equal(
            native_y.view(mx.uint16), folded[:, 0].view(mx.uint16)
        )
    separate = mx.stack([layer(row) for row in dense])
    assert not mx.array_equal(folded.view(mx.uint16), separate.view(mx.uint16)).item()
    set_verify_qmm_armed(True, row_exact=True)
    try:
        assert layer._fast(layer, dense) is None
        _compare(layer, packed, dense, monkeypatch, expected=separate)
    finally:
        set_verify_qmm_armed(False)


def test_planned_plugin_3d_5row_does_not_take_stock_matmul(build_layer, monkeypatch):
    """Only stock layers divert to _stock_matmul on losing decode buckets; a
    planned plugin layer stays on its format route."""
    layer, packed, key = _equivalent_layer("plugin", build_layer, 160, 2560)
    assert not layer._stock
    assert layer._qwen_row_exact is not None
    x = mx.random.normal((5, 1, 2560), key=key).astype(mx.float16)
    _compare(layer, packed, x, monkeypatch)


@pytest.mark.parametrize("change", ["replace", "union", "placeholder"])
def test_writer_mutation_retires_native_replay(change, build_layer, tmp_path):
    layer, packed, key = _projection(build_layer, 160, 2560, 8, mx.float16)
    x = mx.random.normal((2, 1, 2560), key=key).astype(mx.float16)
    before = _stock(x, packed, 8).view(mx.uint16)
    np.testing.assert_array_equal(layer(x).view(mx.uint16), before)
    assert layer._fast is not None
    cloned = copy(layer)
    if change == "placeholder":
        with testing.verified("qwen4_exp"), loading(tmp_path):
            cloned["scales"] = mx.zeros((1,), dtype=packed[1].dtype)
        assert "scales" in cloned._placeholders
        with pytest.raises(ValueError, match="placeholder of scales"):
            mx.eval(cloned(x))
    else:
        zero = mx.zeros_like(packed[1])
        if change == "replace":
            cloned["scales"] = zero
        else:
            cloned |= {"scales": zero}
        expected = _stock(x, (packed[0], zero, packed[2]), 8).view(mx.uint16)
        assert not mx.array_equal(before, expected).item()
        np.testing.assert_array_equal(cloned(x).view(mx.uint16), expected)
    np.testing.assert_array_equal(layer(x).view(mx.uint16), before)


@pytest.mark.parametrize(
    "k,group", [(192, 128), (320, 128), (2560, 4096), (2560, 1 << 30)]
)
def test_wide_reads_the_partial_final_tile_group(k, group, build_layer):
    from tests.test_formats_kernels import _toy
    from tests.test_formats_tile_prefill import _TILE, _tile_toy, _ToyFormat

    n = 16
    source = _TILE.replace("GROUP = 8", f"GROUP = {group}")
    face = _tile_toy(f"partial_{group}", source)
    fmt = _ToyFormat(_toy("partial"), face)
    weights = (129 + mx.arange(n) % 4).astype(mx.uint8)
    codes = mx.broadcast_to(weights[:, None], (n, k))
    layer = build_layer(fmt, 8, 64, k, {"codes": codes, "group_scales": mx.ones((n,))})
    # One nonzero input in each end run exposes an omitted tail even when
    # the cursor covers the whole row in a single logical parameter group.
    x = mx.zeros((2, k), mx.float16)
    x[:, 0] = mx.array([3, 4])
    x[:, -1] = mx.array([5, -2])
    expected = mx.array([[8], [2]], mx.float16) * (weights.astype(mx.float16) - 128)
    for _ in range(2):
        np.testing.assert_array_equal(
            layer(x).view(mx.uint16), expected.view(mx.uint16)
        )
    tag = "format_linear_wide"
    assert any(
        key[0] == tag and face in key and proof.ran
        for key, proof in kernels._PROOFS.items()
    )


@pytest.mark.parametrize("arch", ["unknown", "17s", "applegpu_gXs"])
def test_unrecognized_metal_architecture_uses_existing_format_path(
    arch, build_layer, monkeypatch
):
    from omlx.formats import linear

    qwen, _, key = _equivalent_layer("plugin", build_layer, 16, 2560)
    other, _, _ = _equivalent_layer("plugin", build_layer, 16, 2560, family="llama")
    monkeypatch.setattr(linear, "_metal_architecture", lambda: arch)

    def forbidden(*args):
        pytest.fail("unknown architecture entered the wide traversal")

    monkeypatch.setattr(linear, "_ordinary_tile", forbidden)
    x = mx.random.normal((2, 2560), key=key).astype(mx.float16)
    np.testing.assert_array_equal(qwen(x).view(mx.uint16), other(x).view(mx.uint16))


def test_native_graph_adds_the_loaded_bias(build_layer):
    from omlx.custom_kernels.quant_matmul import fast as native

    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires the native warm dispatcher")
    layer, packed, key = _equivalent_layer("stock", build_layer, 16, 2560)
    layer.bias = mx.arange(16).astype(mx.float16) + 1
    x = mx.random.normal((2, 2560), key=key).astype(mx.float16)
    expected = _stock(x, packed, 8) + layer.bias
    mx.eval(layer(x))
    y = layer._fast(layer, x)
    assert y is not None
    np.testing.assert_array_equal(y.view(mx.uint16), expected.view(mx.uint16))


def test_nested_qwen_text_config_selects_ordinary_traversal(tmp_path, monkeypatch):
    from omlx.patches import format_load

    config = {
        "model_type": "wrapper",
        "text_config": {"model_type": "qwen4_exp"},
        "omlx_quant_formats": {"affine": {}},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    w = mx.zeros((16, 2560), mx.float16)
    arrays = dict(zip(("weight", "scales", "biases"), mx.quantize(w, bits=4)))
    monkeypatch.setattr(format_load, "PORTED", format_load.PORTED | {"wrapper"})
    with testing.verified("wrapper", "qwen4_exp"), loading(tmp_path):
        layer = FormatLinear("affine", 4, 64, 2560, arrays)
    assert layer._qwen_row_exact is not None


def test_native_replay_tracks_worker_replacement_and_eviction():
    from omlx.custom_kernels.quant_matmul import fast as native

    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires the native warm dispatcher")
    x = mx.array([[1.0], [2.0]])
    sig = (x.shape, x.dtype)
    w = object()
    layer, proof = {"weight": w}, kernels.Proof()
    dispatch = native.format_linear_fast(w, None, [w], vars(proof), "weight")
    dispatch.__self__.register(sig)
    results = [object(), object(), object()]
    proof.keep(sig, lambda *_: results[0])
    assert dispatch(layer, x) is results[0]

    def worker():
        assert dispatch(layer, x) is None
        proof.keep(sig, lambda *_: results[2])
        assert dispatch(layer, x) is results[2]

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(worker).result()
        assert dispatch(layer, x) is results[0]
    proof.keep(sig, lambda *_: results[1])
    assert dispatch(layer, x) is results[1]
    old = proof.graphs
    for i in range(kernels.MAX_GRAPHS):
        proof.keep(i, lambda *_: None)
    assert sig in old and sig not in proof.graphs
    assert dispatch(layer, x) is None


def test_wide_masks_the_last_input_group_and_output_pack():
    from omlx.formats import tile_wide
    from tests.test_formats_tile_prefill import _TILE, _tile_toy

    n, k = 13, 320
    face = _tile_toy("partial_guard", _TILE.replace("GROUP = 8", "GROUP = 128"))
    # Reserve the full grid footprint, including an extra group following K.
    # An illegal read or write within that padding gives a defined witness.
    x = mx.ones((384,), mx.float16)
    codes = mx.full((16 * k + 64,), 129, mx.uint8)
    scales = mx.ones((16,), mx.float32)
    (got,) = tile_wide.kernel(face)(
        inputs=[x, codes, scales],
        template=[
            ("T", mx.float16),
            ("K", k),
            ("N", n),
            ("V", 1),
            ("PARTIAL_M", False),
        ],
        grid=(32, 4, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(16,)],
        output_dtypes=[mx.float16],
        init_value=-31,
    )
    np.testing.assert_array_equal(np.asarray(got), [k] * n + [-31] * (16 - n))


def test_planless_stock_affine_row_exact_matches_serial_decode(
    build_layer, monkeypatch
):
    # K 128 at 8 bits is MLX's qmv_quad: the affine format has no dot face,
    # so the layer has no plan. An armed verification step still keeps MLX's
    # own matmul, a row at a time (as row_exact_qmv does), never the decoded
    # matrix.
    for dtype in (mx.bfloat16, mx.float16):
        w = mx.random.normal((256, 128), key=mx.random.key(211)).astype(dtype)
        packed = mx.quantize(w, group_size=128, bits=8)
        names = ("weight", "scales", "biases")
        layer = build_layer("affine", 8, 128, 128, dict(zip(names, packed)))
        assert linear._plan(layer) is None
        object.__setattr__(layer, "_qwen_row_exact", lambda: True)

        def trap(self, *_args, **_kwargs):
            raise AssertionError("a row-exact stock layer decoded its matrix")

        monkeypatch.setattr(FormatLinear, "_decoded", trap)
        for shape in ((128,), (4, 128), (1, 3, 128)):
            x = mx.random.normal(shape, key=mx.random.key(7)).astype(dtype)
            rows = x.reshape(-1, 128)
            want = mx.concatenate(
                [
                    mx.quantized_matmul(
                        rows[r : r + 1],
                        *packed,
                        transpose=True,
                        bits=8,
                        group_size=128,
                    )
                    for r in range(rows.shape[0])
                ],
                axis=0,
            ).reshape(*shape[:-1], 256)
            y = layer(x)
            assert y.shape == (*shape[:-1], 256)
            assert mx.array_equal(y, want).item()


def test_plugin_qmm_returns_before_allocating_routes(build_layer):
    # 513 rows at N 512 / K 2560 traverse one part (qmm), a route no plugin
    # face serves: the call returns before it keeps a routes slot.
    layer, _, _ = _affine_plugin(build_layer, 512, 2560, 4, mx.float16)
    got = layer(mx.zeros((513, 2560), mx.float16))
    mx.eval(got)
    assert vars(layer)[linear._KEPT].routes == {}


def test_plugin_rows_traversal_keeps_the_rows_route(build_layer, monkeypatch):
    # On a g13/g14 GPU two rows traverse the rows route; a plugin layer must
    # keep it (the shared-rows launch whose rows have their one-row bits).
    from omlx.formats import linear as linear_mod

    layer, _, _ = _affine_plugin(build_layer, 64, 2560, 4, mx.float16)
    monkeypatch.setattr(linear_mod, "_metal_architecture", lambda: "applegpu_g13s")
    x = mx.random.normal((2, 2560)).astype(mx.float16)
    serial = mx.concatenate([layer(x[i : i + 1]) for i in range(2)])
    got = layer(x)
    mx.eval(got, serial)
    np.testing.assert_array_equal(
        np.asarray(got.view(mx.uint16)), np.asarray(serial.view(mx.uint16))
    )
    plan = vars(layer)[linear._KEPT].plan
    assert any(
        isinstance(key, tuple) and key and key[0] == "ordinary_rows"
        for key in plan.proof.graphs
    )


def test_planless_stock_affine_row_exact_empty_input_avoids_decode(
    build_layer, monkeypatch
):
    w = mx.random.normal((256, 128), key=mx.random.key(311)).astype(mx.float16)
    packed = mx.quantize(w, group_size=128, bits=8)
    names = ("weight", "scales", "biases")
    layer = build_layer("affine", 8, 128, 128, dict(zip(names, packed)))
    object.__setattr__(layer, "_qwen_row_exact", lambda: True)

    def trap(self, *_args, **_kwargs):
        raise AssertionError("an empty row-exact stock call decoded its matrix")

    monkeypatch.setattr(FormatLinear, "_decoded", trap)
    y = layer(mx.zeros((0, 128), mx.float16))
    assert y.shape == (0, 256) and y.dtype == mx.float16


def test_plugin_format_linear_empty_input_avoids_decode(build_layer, monkeypatch):
    layer, _, _ = _affine_plugin(build_layer, 256, 128, 4, mx.float16)
    monkeypatch.setattr(
        FormatLinear,
        "_decoded",
        lambda self, *a, **k: pytest.fail("an empty plugin call decoded"),
    )
    y = layer(mx.zeros((0, 128), mx.float16))
    assert y.shape == (0, 256) and y.dtype == mx.float16


def test_planless_stock_affine_row_exact_above_max_rows_matches_serial_decode(
    build_layer, monkeypatch
):
    # Armed verification on a plan-less qmv_quad stock layer must stay
    # row-exact even above MAX_ROWS (16), never taking _prefill or _decoded,
    # and without building a discarded full-batch _stock_matmul first.
    monkeypatch.setattr(
        FormatLinear,
        "_decoded",
        lambda self, *a, **k: pytest.fail("row-exact stock call decoded"),
    )
    monkeypatch.setattr(
        FormatLinear,
        "_prefill",
        lambda self, *a, **k: pytest.fail("row-exact stock call took _prefill"),
    )
    calls = []
    orig_stock = FormatLinear._stock_matmul

    def counted(self, x):
        calls.append(x.shape)
        return orig_stock(self, x)

    monkeypatch.setattr(FormatLinear, "_stock_matmul", counted)
    for dtype in (mx.bfloat16, mx.float16):
        w = mx.random.normal((256, 128), key=mx.random.key(223)).astype(dtype)
        packed = mx.quantize(w, group_size=128, bits=8)
        names = ("weight", "scales", "biases")
        layer = build_layer("affine", 8, 128, 128, dict(zip(names, packed)))
        assert linear._plan(layer) is None
        object.__setattr__(layer, "_qwen_row_exact", lambda: True)

        for rows_count in (4, 17):
            calls.clear()
            x = mx.random.normal((rows_count, 128), key=mx.random.key(19)).astype(dtype)
            want = mx.concatenate(
                [
                    mx.quantized_matmul(
                        x[r : r + 1],
                        *packed,
                        transpose=True,
                        bits=8,
                        group_size=128,
                    )
                    for r in range(rows_count)
                ],
                axis=0,
            )
            y = layer(x)
            assert y.shape == (rows_count, 256)
            assert mx.array_equal(y, want).item()
            assert len(calls) == rows_count


def test_planless_stock_affine_short_batch_uses_stock_matmul_up_to_max_rows(
    build_layer, monkeypatch
):
    # Unarmed calls up to MAX_ROWS (16) on a plan-less stock layer keep MLX's
    # quantized_matmul instead of falling through to _decoded.
    w = mx.random.normal((256, 128), key=mx.random.key(227)).astype(mx.bfloat16)
    packed = mx.quantize(w, group_size=128, bits=8)
    names = ("weight", "scales", "biases")
    layer = build_layer("affine", 8, 128, 128, dict(zip(names, packed)))
    assert linear._plan(layer) is None
    object.__setattr__(layer, "_qwen_row_exact", None)

    monkeypatch.setattr(
        FormatLinear,
        "_decoded",
        lambda self, *a, **k: pytest.fail("short plan-less stock call decoded"),
    )
    monkeypatch.setattr(
        FormatLinear,
        "_prefill",
        lambda self, *a, **k: pytest.fail("short plan-less stock call took _prefill"),
    )
    for rows_count in (12, linear.MAX_ROWS):
        x = mx.random.normal((rows_count, 128), key=mx.random.key(23)).astype(
            mx.bfloat16
        )
        want = mx.quantized_matmul(x, *packed, transpose=True, bits=8, group_size=128)
        y = layer(x)
        assert y.shape == (rows_count, 256)
        assert mx.array_equal(y.view(mx.uint16), want.view(mx.uint16)).item()


def test_planless_plugin_skips_stock_matmul_under_row_exact(build_layer, monkeypatch):
    # A plugin format with no dot face never calls _stock_matmul, even under
    # armed verification; and rows == MAX_ROWS stays on the non-prefill path.
    layer, _, _ = _affine_plugin(build_layer, 128, 64, 4, mx.bfloat16)
    assert linear._plan(layer) is None
    object.__setattr__(layer, "_qwen_row_exact", lambda: True)
    calls = []
    orig_stock = FormatLinear._stock_matmul

    def counted(self, x):
        calls.append(x.shape)
        return orig_stock(self, x)

    monkeypatch.setattr(FormatLinear, "_stock_matmul", counted)
    monkeypatch.setattr(
        FormatLinear,
        "_prefill",
        lambda self, *a, **k: pytest.fail("rows == MAX_ROWS took _prefill"),
    )
    for rows_count in (8, linear.MAX_ROWS):
        y = layer(mx.zeros((rows_count, 64), mx.bfloat16))
        mx.eval(y)
        assert y.shape == (rows_count, 128)
    assert calls == []


def test_planless_stock_affine_unarmed_above_max_rows_skips_stock_matmul(
    build_layer, monkeypatch
):
    # Unarmed calls above MAX_ROWS on plan-less stock affine take _prefill /
    # _decoded, not _stock_matmul (kills mutating rows <= MAX_ROWS to True).
    w = mx.random.normal((256, 128), key=mx.random.key(229)).astype(mx.bfloat16)
    packed = mx.quantize(w, group_size=128, bits=8)
    names = ("weight", "scales", "biases")
    layer = build_layer("affine", 8, 128, 128, dict(zip(names, packed)))
    assert linear._plan(layer) is None
    object.__setattr__(layer, "_qwen_row_exact", None)

    monkeypatch.setattr(
        FormatLinear,
        "_stock_matmul",
        lambda self, *a, **k: pytest.fail("unarmed rows > MAX_ROWS took _stock_matmul"),
    )
    x = mx.random.normal((linear.MAX_ROWS + 1, 128), key=mx.random.key(29)).astype(
        mx.bfloat16
    )
    y = layer(x)
    mx.eval(y)
    assert y.shape == (linear.MAX_ROWS + 1, 256)


def test_planless_plugin_row_exact_above_max_rows_skips_prefill_and_matches_serial(
    build_layer, monkeypatch
):
    class _PlanlessPlugin(_StockLayoutPlugin):
        mode = "planless_control"

        def dot(self, layer, k, n):
            return None

    k, n, rows = 512, 128, linear.MAX_ROWS + 1
    keys = mx.random.split(mx.random.key(707), 2)
    w_half = mx.random.normal((n, k // 2), key=keys[0]).astype(mx.bfloat16)
    w = mx.concatenate([w_half, w_half], axis=1)
    packed = mx.quantize(w, group_size=64, bits=4)
    layer = build_layer(
        _PlanlessPlugin(),
        4,
        64,
        k,
        dict(zip(("weight", "scales", "biases"), packed)),
    )
    object.__setattr__(layer, "_qwen_row_exact", lambda: True)
    monkeypatch.setattr(
        FormatLinear,
        "_prefill",
        lambda *arguments, **keywords: pytest.fail(
            "a row-exact plugin call took the dense prefill"
        ),
    )
    half = mx.random.normal((rows, k // 2), key=keys[1]).astype(mx.bfloat16)
    x = mx.concatenate([half, -half], axis=1)
    y = layer(x)
    serial = mx.concatenate([layer(x[r : r + 1]) for r in range(rows)], axis=0)
    batched = x @ layer._decoded().astype(mx.bfloat16).T
    assert not mx.array_equal(
        batched.view(mx.uint16), serial.view(mx.uint16)
    ).item(), "fixture must distinguish batched matmul from 1-row serial matmul"
    assert mx.array_equal(y.view(mx.uint16), serial.view(mx.uint16)).item()


def test_decision_backbone_takes_format_embedding_and_linear_heads(tmp_path):
    import json

    from omlx.formats import testing as fmt_testing
    from omlx.formats.linear import FormatEmbedding
    from omlx.models.decision import DecisionBackbone
    from omlx.patches.format_load import loading

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "omlx_quant_formats": {"affine": {}}})
    )
    mx.random.seed(83)
    embedding = (mx.random.normal((16, 256)) * 0.1).astype(mx.float16)
    arrays = dict(zip(("weight", "scales", "biases"), mx.quantize(embedding, bits=4)))
    with fmt_testing.verified("qwen4_exp"), loading(tmp_path):
        head = FormatEmbedding("affine", 4, 64, 256, arrays)
    backbone = DecisionBackbone.__new__(DecisionBackbone)
    backbone._lm_head = head
    ids = mx.array([0, 3, 15])
    rows = backbone.output_rows(ids)
    mx.eval(rows)
    want = head(ids)
    mx.eval(want)
    assert mx.array_equal(rows.view(mx.uint16), want.view(mx.uint16))
    ids_2d = mx.array([[0, 3], [5, 15]])
    rows_2d = backbone.output_rows(ids_2d)
    mx.eval(rows_2d)
    assert rows_2d.shape == (2, 2, 256)
    assert mx.array_equal(rows_2d.view(mx.uint16), head(ids_2d).view(mx.uint16))
    hidden = (mx.random.normal((1, 4, 256)) * 0.5).astype(mx.float16)
    logits = backbone.logits(hidden)
    mx.eval(logits)
    expected = head.as_linear(hidden)
    mx.eval(expected)
    assert mx.array_equal(logits.view(mx.uint16), expected.view(mx.uint16))


def test_decision_backbone_format_linear_head(tmp_path, build_layer):
    """DecisionBackbone must decode FormatLinear heads through the format
    interface (affine parity, custom plugin codes_name, and placeholder
    refusal), never hand-reading .weight/.scales."""
    import json

    from omlx.formats import QuantFormat
    from omlx.formats import testing as fmt_testing
    from omlx.formats.linear import FormatLinear
    from omlx.models.decision import DecisionBackbone
    from omlx.patches.format_load import loading

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "omlx_quant_formats": {"affine": {}}})
    )
    mx.random.seed(89)
    weight = (mx.random.normal((16, 256)) * 0.1).astype(mx.float16)
    packed = mx.quantize(weight, bits=4, group_size=64)
    arrays = dict(zip(("weight", "scales", "biases"), packed))
    with fmt_testing.verified("qwen4_exp"), loading(tmp_path):
        affine_head = FormatLinear("affine", 4, 64, 256, arrays)

    backbone = DecisionBackbone.__new__(DecisionBackbone)
    backbone._lm_head = affine_head
    ids = mx.array([0, 3, 15])
    rows = backbone.output_rows(ids)
    mx.eval(rows)
    want = mx.dequantize(
        packed[0][ids],
        packed[1][ids],
        packed[2][ids],
        group_size=64,
        bits=4,
        mode="affine",
    )
    mx.eval(want)
    assert mx.array_equal(rows.view(mx.uint16), want.view(mx.uint16))
    hidden = (mx.random.normal((1, 4, 256)) * 0.5).astype(mx.float16)
    logits = backbone.logits(hidden)
    mx.eval(logits)
    expected = affine_head(hidden)
    mx.eval(expected)
    assert mx.array_equal(logits.view(mx.uint16), expected.view(mx.uint16))

    # Custom plugin with non-standard codes_name (no .weight or .scales).
    class _CustomCodesPlugin(QuantFormat):
        mode = "custom_codes_decision_test"
        codes_name = "payload"
        tensors = ("meta_s", "meta_b")
        bits = (4,)
        group_sizes = (64,)

        def layout(self, k, n, bits, group_size):
            return {
                "payload": ((n, k * bits // 32), mx.uint32),
                "meta_s": ((n, k // group_size), None),
                "meta_b": ((n, k // group_size), None),
            }

        def decode(self, tensors, bits, group_size):
            return mx.dequantize(
                tensors["payload"],
                tensors["meta_s"],
                tensors["meta_b"],
                group_size=group_size,
                bits=bits,
                mode="affine",
            )

    plugin_head = build_layer(
        _CustomCodesPlugin(),
        4,
        64,
        256,
        {"payload": packed[0], "meta_s": packed[1], "meta_b": packed[2]},
    )
    backbone._lm_head = plugin_head
    plugin_rows = backbone.output_rows(ids)
    mx.eval(plugin_rows)
    assert mx.array_equal(plugin_rows.view(mx.uint16), want.view(mx.uint16))

    # Placeholders must be refused instead of decoding zeros.
    with fmt_testing.verified("qwen4_exp"), loading(tmp_path):
        placeholder_head = FormatLinear(
            "affine",
            4,
            64,
            256,
            {k: mx.zeros_like(v) for k, v in arrays.items()},
            placeholders=True,
        )
    backbone._lm_head = placeholder_head
    with pytest.raises(ValueError, match="placeholder"):
        backbone.output_rows(ids)


def test_decision_backbone_non_stackable_head_selects_rows(build_layer):
    from omlx import formats
    from omlx.formats import QuantFormat
    from omlx.models.decision import DecisionBackbone

    class _RowCentered(QuantFormat):
        mode = "row_centered_decision_test"
        codes_name = "payload"
        tensors = ("meta_s", "meta_b")
        bits = (4,)
        group_sizes = (64,)

        def layout(self, k, n, bits, group_size):
            return None

        def decode(self, tensors, bits, group_size):
            decoded = mx.dequantize(
                tensors["payload"],
                tensors["meta_s"],
                tensors["meta_b"],
                group_size=group_size,
                bits=bits,
                mode="affine",
            )
            return decoded - decoded.mean(axis=-2, keepdims=True)

    mx.random.seed(97)
    weight = (mx.random.normal((16, 256)) * 0.1).astype(mx.float16)
    packed = mx.quantize(weight, bits=4, group_size=64)
    arrays = {"payload": packed[0], "meta_s": packed[1], "meta_b": packed[2]}
    head = build_layer(_RowCentered(), 4, 64, 256, arrays)
    assert not formats.stackable(head.fmt, head.input_dims, head.output_dims, 4, 64)
    backbone = DecisionBackbone.__new__(DecisionBackbone)
    backbone._lm_head = head
    ids = mx.array([0, 3, 15])
    rows = backbone.output_rows(ids)
    mx.eval(rows)
    want = head._decoded()[ids]
    mx.eval(want)
    assert rows.shape == (3, 256)
    assert mx.array_equal(rows.view(mx.uint16), want.view(mx.uint16))


def test_decision_backbone_two_dimensional_ids(build_layer):
    from omlx.formats import QuantFormat
    from omlx.models.decision import DecisionBackbone

    seen = []

    class _Recording(QuantFormat):
        mode = "recording_decision_test"
        codes_name = "payload"
        tensors = ("meta_s", "meta_b")
        bits = (4,)
        group_sizes = (64,)

        def layout(self, k, n, bits, group_size):
            return {
                "payload": ((n, k * bits // 32), mx.uint32),
                "meta_s": ((n, k // group_size), None),
                "meta_b": ((n, k // group_size), None),
            }

        def decode(self, tensors, bits, group_size):
            seen.append(tuple(tensors["payload"].shape))
            return mx.dequantize(
                tensors["payload"],
                tensors["meta_s"],
                tensors["meta_b"],
                group_size=group_size,
                bits=bits,
                mode="affine",
            )

    mx.random.seed(101)
    weight = (mx.random.normal((16, 256)) * 0.1).astype(mx.float16)
    packed = mx.quantize(weight, bits=4, group_size=64)
    arrays = {"payload": packed[0], "meta_s": packed[1], "meta_b": packed[2]}
    head = build_layer(_Recording(), 4, 64, 256, arrays)
    backbone = DecisionBackbone.__new__(DecisionBackbone)
    backbone._lm_head = head
    scalar = backbone.output_rows(mx.array(3))
    mx.eval(scalar)
    matrix_ids = mx.array([[0, 1], [3, 15]])
    rows = backbone.output_rows(matrix_ids)
    mx.eval(rows)
    assert scalar.shape == (256,)
    assert rows.shape == (2, 2, 256)
    assert seen == [
        (1, packed[0].shape[1]),
        (4, packed[0].shape[1]),
    ], seen
    assert mx.array_equal(
        rows.view(mx.uint16),
        head._decoded(matrix_ids.reshape(-1)).reshape(2, 2, 256).view(mx.uint16),
    )
    assert mx.array_equal(
        scalar.view(mx.uint16),
        head._decoded(mx.array([3])).reshape(256).view(mx.uint16),
    )
