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
from omlx.formats import dot, kernels, testing, tile
from omlx.formats.affine import AffineFormat
from omlx.formats.linear import FormatLinear
from omlx.patches.format_load import loading
from omlx.patches.qwen35_verify_qmm import set_verify_qmm_armed

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture
def build_layer(tmp_path):
    def build(fmt, bits, group, k, tensors, family="qwen4_exp"):
        mode = fmt if isinstance(fmt, str) else fmt.mode
        config = {"model_type": family, "omlx_quant_formats": {mode: {}}}
        (tmp_path / "config.json").write_text(json.dumps(config))
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
@pytest.mark.parametrize("bits", [4, 8])
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
def test_gapped_single_row_matrices_use_each_formats_dot(
    mode, build_layer, monkeypatch
):
    k, n = 2560, 48
    layer, packed, key = _equivalent_layer(mode, build_layer, n, k)
    # Three matrices exercise the general layout selection beyond DecodePair.
    gapped = _gapped_rows(1, k, key, matrices=3)
    dense = mx.contiguous(gapped)
    face = dot(layer, k, n)
    operands = [layer[name] for name in (formats.tensor_names(layer)[0], *face.tensors)]
    singles = mx.stack([testing.dot_rows(face, operands, x[0], n) for x in gapped])
    singles = singles[:, None, :]
    stock = _stock(gapped, packed, 8)
    stock_singles = mx.stack([_stock(x, packed, 8) for x in gapped])
    np.testing.assert_array_equal(stock_singles.view(mx.uint16), stock.view(mx.uint16))
    folded = _stock(dense, packed, 8)
    assert not mx.array_equal(stock.view(mx.uint16), folded.view(mx.uint16)).item()
    # Only the centered-byte plugin multiplies x by (q - 128) in its Dot.
    # The affine Dot accumulates q*x and applies the bias separately.
    if mode == "plugin":
        assert not mx.array_equal(singles.view(mx.uint16), stock.view(mx.uint16)).item()
    else:
        np.testing.assert_array_equal(singles.view(mx.uint16), stock.view(mx.uint16))
    _compare(layer, packed, dense, monkeypatch)
    _compare(layer, packed, gapped, monkeypatch, expected=singles)
    _compare(layer, packed, dense, monkeypatch)


@pytest.mark.parametrize("mode", ["stock", "plugin"])
def test_batched_format_calls_keep_the_stock_traversal_on_warm_views(
    mode, build_layer, monkeypatch
):
    rows, k, n = 5, 2560, 48
    layer, packed, key = _equivalent_layer(mode, build_layer, n, k)
    gapped = _gapped_rows(rows, k, key)
    dense = mx.contiguous(gapped)
    for x in (dense, gapped, dense):
        _compare(layer, packed, x, monkeypatch)


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
def test_ordinary_two_request_decode_replays_one_graph_for_live_layouts(
    mode, dtype, build_layer, monkeypatch
):
    from omlx.custom_kernels.quant_matmul import fast as native

    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires the native warm dispatcher")
    k, n = 2560, 512
    layer, packed, key = _equivalent_layer(mode, build_layer, n, k, dtype=dtype)
    if mode == "stock":
        assert layer.scales is packed[1] and layer.biases is packed[2]
    gapped = mx.repeat(_gapped_rows(1, k, key).astype(dtype), 2, axis=0)[::2]
    dense = mx.contiguous(gapped)
    folded = _stock(dense, packed, 8)
    separate = mx.stack([layer(row) for row in gapped])
    assert not mx.array_equal(folded.view(mx.uint16), separate.view(mx.uint16)).item()
    _compare(layer, packed, dense, monkeypatch, expected=folded)
    assert layer._fast is not None, "two-request layout has no native replay"

    broadcast = mx.broadcast_to(gapped[:1], gapped.shape)
    column = mx.repeat(dense, 2, axis=-1)[..., ::2]
    layouts = [(dense, folded), (gapped, separate), (dense, folded)]
    for x in (broadcast, column):
        # Stock copies these layouts before folding the two singleton matrices.
        stock = _stock(x, packed, 8)
        contiguous = _stock(mx.contiguous(x), packed, 8)
        singletons = mx.stack([_stock(row, packed, 8) for row in x])
        np.testing.assert_array_equal(stock.view(mx.uint16), contiguous.view(mx.uint16))
        assert not mx.array_equal(
            stock.view(mx.uint16), singletons.view(mx.uint16)
        ).item()
        layouts.append((x, stock))

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
    for x, expected in (
        (ordinary, folded[:, 0]),
        (gapped, separate),
        (ordinary, folded[:, 0]),
    ):
        _compare(layer, packed, x, monkeypatch, expected=expected)
        native_y = layer._fast(layer, x)
        assert native_y is not None, "warm call kept a different plan's dispatcher"
        np.testing.assert_array_equal(
            native_y.view(mx.uint16), expected.view(mx.uint16)
        )
    set_verify_qmm_armed(True, row_exact=True)
    try:
        assert layer._fast(layer, dense) is None
        _compare(layer, packed, dense, monkeypatch, expected=separate)
    finally:
        set_verify_qmm_armed(False)


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
@pytest.mark.parametrize("pair", [False, True])
def test_wide_reads_the_partial_final_tile_group(k, group, pair, build_layer):
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
    if pair:
        x = x[:, None, :]
        expected = expected[:, None, :]
    for _ in range(2):
        np.testing.assert_array_equal(
            layer(x).view(mx.uint16), expected.view(mx.uint16)
        )
    tag = "format_linear_decode_pair" if pair else "format_linear_wide"
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


def test_decode_pair_leaves_partial_output_pack_to_wide(build_layer, monkeypatch):
    from omlx.formats import linear

    layer, packed, key = _equivalent_layer("stock", build_layer, 13, 2560)
    x = mx.random.normal((2, 1, 2560), key=key).astype(mx.float16)

    def refuse_pair(*args):
        pytest.fail("two-request Dot requires complete output packs")

    monkeypatch.setattr(linear, "_decode_pair", refuse_pair)
    _compare(layer, packed, x, monkeypatch)


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


def test_nested_qwen_text_config_selects_ordinary_traversal(tmp_path):
    config = {
        "model_type": "wrapper",
        "text_config": {"model_type": "qwen4_exp"},
        "omlx_quant_formats": {"affine": {}},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    w = mx.zeros((16, 2560), mx.float16)
    arrays = dict(zip(("weight", "scales", "biases"), mx.quantize(w, bits=4)))
    with testing.verified("wrapper", "qwen4_exp"), loading(tmp_path):
        layer = FormatLinear("affine", 4, 64, 2560, arrays)
    assert layer._qwen_row_exact is not None


def test_decode_pair_counts_both_faces_tables(build_layer):
    from dataclasses import replace

    from omlx.formats import Table, linear

    layer, _, _ = _equivalent_layer("plugin", build_layer, 16, 2560)
    layer.fmt.dot_face = replace(
        layer.fmt.dot_face, tables=(Table("dot_lut", mx.float32, (0.0,) * 1024),)
    )
    layer.fmt.tile_face = replace(
        layer.fmt.tile_face, tables=(Table("tile_lut", mx.uint8, (1,)),)
    )
    plan = linear._plan(layer)
    assert kernels.table_bytes(plan.face, layer.fmt.tile_face) == 4097
    assert linear._decode_pair(layer, plan) is None


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


def test_decode_pair_binds_strided_static_operands_once(build_layer, monkeypatch):
    from omlx.formats import linear

    n, k = 16, 2560
    _, packed, key = _projection(build_layer, n, k, 8, mx.float16)
    strided = tuple(mx.repeat(a, 2, axis=-1)[..., ::2] for a in packed)
    names = ("weight", "scales", "biases")
    layer = build_layer("affine", 8, 64, k, dict(zip(names, strided)))
    x = mx.random.normal((2, 1, k), key=key).astype(mx.float16)
    want = _stock(x, packed, 8)
    mx.eval(want)
    contiguous = mx.contiguous
    seen = []

    def observe(array, *args, **kwargs):
        for name in names:
            if array is layer[name]:
                seen.append(name)
        return contiguous(array, *args, **kwargs)

    monkeypatch.setattr(mx, "contiguous", observe)
    for _ in range(2):
        np.testing.assert_array_equal(layer(x).view(mx.uint16), want.view(mx.uint16))
    assert sorted(seen) == sorted(names), "bind each strided operand exactly once"
    kept = vars(layer)[linear._KEPT]
    assert kept[3]["decode_pair"] is not None
    assert layer._fast is not None

    replacement = mx.zeros_like(strided[1])
    layer["scales"] = replacement
    want = _stock(x, (packed[0], replacement, packed[2]), 8)
    for _ in range(2):
        np.testing.assert_array_equal(layer(x).view(mx.uint16), want.view(mx.uint16))
    assert sorted(seen) == sorted(2 * names)


def test_decode_pair_first_binds_inside_a_compiled_layer_call(build_layer):
    n, k = 16, 2560
    layer, packed, key = _projection(build_layer, n, k, 8, mx.float16)
    x = mx.random.normal((2, 1, k), key=key).astype(mx.float16)
    step = mx.compile(lambda value: layer(value), inputs=layer.state)
    want = _stock(x, packed, 8)
    for call in (step, layer, step):
        np.testing.assert_array_equal(call(x).view(mx.uint16), want.view(mx.uint16))
    replacement = mx.zeros_like(packed[2])
    layer["biases"] = replacement
    want = _stock(x, (packed[0], packed[1], replacement), 8)
    for call in (step, layer):
        np.testing.assert_array_equal(call(x).view(mx.uint16), want.view(mx.uint16))
