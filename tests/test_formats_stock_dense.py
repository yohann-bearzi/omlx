# SPDX-License-Identifier: Apache-2.0
"""Ordinary Qwen dense projections compared with MLX's public affine operation."""

import gc
import json
import logging
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from threading import Barrier

import mlx.core as mx
import numpy as np
import pytest

from omlx import formats
from omlx.formats import kernels, linear, testing
from omlx.formats.affine import AffineFormat
from omlx.formats.linear import FormatLinear
from omlx.patches.format_load import loading

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


def _projection(build, n, k, bits, dtype, *, paired_halves=False, stock=True):
    keys = mx.random.split(mx.random.key(161 + n + bits))
    width = k // 2 if paired_halves else k
    w = mx.random.normal((n, width), key=keys[0]).astype(dtype)
    if paired_halves:
        w = mx.concatenate([w, w], axis=1)
    packed = mx.quantize(w, group_size=64, bits=bits)
    names = ("weight", "scales", "biases")
    layer = build(
        "affine" if stock else _AffinePlugin(),
        bits,
        64,
        k,
        dict(zip(names, packed)),
    )
    return layer, packed, keys[1]


def _stock(x, packed, bits):
    return mx.quantized_matmul(x, *packed, transpose=True, bits=bits, group_size=64)


def _one_expert_of(packed, bits, group_size, x):
    """The one-expert gathered matmul's bits over MLX's own tensors: the
    arithmetic the format's dense tile runs (pinned in
    tests/test_formats_tile_dense.py)."""
    from mlx.nn import QuantizedLinear

    from tests.test_formats_tile_dense import _one_expert

    reference = QuantizedLinear(x.shape[-1], packed[0].shape[0], bias=False)
    reference.weight, reference.scales, reference.biases = packed
    reference.group_size, reference.bits = group_size, bits
    return _one_expert(reference, x)


def _compare(layer, packed, x, monkeypatch, *, expected=None, mlx_route=False):
    """Compare ``layer(x)`` with MLX's own qmm bits, forbidding qmm/decode
    unless the route under test is MLX's (a stock layer's unmeasured
    2..16-row bucket, lead H4)."""
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
    if not mlx_route:
        assert not calls, f"dense format layer used {calls}"
    return got


def _serial_stock(x, packed, bits):
    """Per-row stock qmv oracle for msg_tile C2 comparisons on MLX 0.32.5 NAX."""
    rows = x.shape[0]
    return mx.concatenate(
        [_stock(x[r : r + 1], packed, bits) for r in range(rows)],
        axis=0,
    )


def _mlx0325_nax_replaces_qmv_wide(rows: int, bits: int) -> bool:
    """MLX 0.32.5 routes M>=4 (4-bit) and M>=6 (3/8-bit) on NAX to qmm_t_nax
    instead of qmv_wide, changing the K-reduction order of mx.quantized_matmul."""
    from omlx.custom_kernels.nax import is_nax_available

    if not is_nax_available():
        return False
    return (bits == 4 and rows >= 4) or (bits in (3, 8) and rows >= 6)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 7])
def test_wide_rows_match_the_stock_oracle_on_a_plugin_face(
    rows, bits, dtype, build_layer, monkeypatch
):
    # A stock layer's 2..16 rows take MLX by DATA (lead H4); the format's
    # wide route serves an unmeasured plugin face over the same matrix and
    # bits, and is bitwise to MLX's qmv_wide oracle. Clear lookup so
    # _AffinePlugin's reused affine face_id tests the unmeasured plugin wide
    # fallback rather than a measured stock msg_tile entry.
    from omlx.formats import tuning

    monkeypatch.setattr(tuning, "lookup", lambda *a, **kw: None)
    k, n = 2560, 512
    layer, packed, key = _projection(
        build_layer, n, k, bits, dtype, paired_halves=True, stock=False
    )
    if _mlx0325_nax_replaces_qmv_wide(rows, bits):
        # MLX 0.32.5 runs qmm_t_nax here rather than qmv_wide; verify the
        # wide route runs without MLX fallbacks and meets the C2 FP32 bound.
        x = mx.random.normal((rows, k), key=key).astype(dtype)
        want = _stock(x, packed, bits)
        mx.eval(want)
        calls = []
        with monkeypatch.context() as patch:
            for name in ("quantized_matmul", "gather_qmm", "dequantize"):
                orig = getattr(mx, name)
                patch.setattr(
                    mx,
                    name,
                    lambda *a, _n=name, _f=orig, **kw: (calls.append(_n), _f(*a, **kw))[
                        1
                    ],
                )
            got = layer(x)
            mx.eval(got)
        assert not calls, f"dense format layer used {calls}"
        assert got.shape == want.shape and got.dtype == want.dtype
        _c2_ok(got, _serial_stock(x, packed, bits), x, packed, bits)
        return
    # Equal weight halves with opposite inputs expose differences in the
    # reduction order even when low-precision outputs usually round alike.
    half = mx.random.normal((rows, k // 2), key=key).astype(dtype)
    x = mx.concatenate([half, -half], axis=1)
    _compare(layer, packed, x, monkeypatch)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_dense_rows_above_sixteen_meet_the_fp32_reference(
    bits, dtype, build_layer, monkeypatch
):
    # Above MAX_ROWS the schedule's dense tile reorders the accumulation
    # (rule 33): the fp32 reference from the same weights decides (C2).
    k, n, rows = 2560, 512, 17
    layer, packed, key = _projection(
        build_layer, n, k, bits, dtype, paired_halves=True, stock=False
    )
    x = mx.random.normal((rows, k), key=key).astype(dtype)
    got = layer(x)
    stock = _stock(x, packed, bits)
    mx.eval(got, stock)
    weight = mx.dequantize(
        packed[0], packed[1], packed[2], group_size=64, bits=bits
    ).astype(mx.float32)
    reference = x.astype(mx.float32) @ weight.T

    def rel(array):
        return float(
            mx.sqrt(mx.sum((array.astype(mx.float32) - reference) ** 2))
            / mx.sqrt(mx.sum(reference**2))
        )

    assert rel(got) <= 1.05 * rel(stock) + 1e-6, (rel(got), rel(stock))


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


def test_a_plugin_layer_takes_the_wide_route(build_layer):
    """A plugin layer's 5-row call takes the format's wide route: MLX's
    batched bits, not a serial one-row decode's, and the dense proof ran
    (Rule 33 retired the armed row-exact half; a stock layer's unmeasured
    5-row bucket is MLX's by DATA, lead H4)."""
    k, n, rows = 2560, 160, 5
    layer, packed, key = _equivalent_layer(
        "plugin", build_layer, n, k, dtype=mx.float16
    )
    half = mx.random.normal((rows, k // 2), key=key).astype(mx.float16)
    x = mx.concatenate([half, -half], axis=1)
    wide = mx.quantized_matmul(x, *packed, transpose=True, bits=8, group_size=64)
    serial = mx.concatenate([layer(x[i : i + 1]) for i in range(rows)])
    wide_bits = np.asarray(wide.view(mx.uint16))
    serial_bits = np.asarray(serial.view(mx.uint16))
    assert np.any(wide_bits != serial_bits), "fixture must distinguish the traversals"
    got = np.asarray(layer(x).view(mx.uint16))
    np.testing.assert_array_equal(got, wide_bits)
    kept = vars(layer)[linear._KEPT]
    plan = kept.dense
    tag = "format_linear_tile_nax" if linear._nax_host() else "format_linear_tile_steel"
    assert any(
        name[0] == tag and plan.face in name and proof.ran
        for name, proof in kernels._PROOFS.items()
    )


@pytest.mark.parametrize("mode", ["plugin"])
def test_a_warm_route_replays_until_its_proof_fails(mode, build_layer, monkeypatch):
    from omlx.formats import linear, tile_dense

    k, n = 2560, 160
    layer, packed, key = _equivalent_layer(mode, build_layer, n, k)
    half = mx.random.normal((1, 2, k // 2), key=key).astype(mx.float16)
    x = mx.concatenate([half, -half], axis=-1)
    wide = _stock(x, packed, 8)
    serial = mx.stack([layer(row) for row in x[0]])[None]
    assert not mx.array_equal(wide.view(mx.uint16), serial.view(mx.uint16)).item()
    cold = []
    project = tile_dense._project_cold

    def record(plan, z, signature):
        cold.append(z.shape)
        return project(plan, z, signature)

    monkeypatch.setattr(tile_dense, "_project_cold", record)
    _compare(layer, packed, x, monkeypatch)
    assert cold, "the first call built no route"
    cold.clear()
    _compare(layer, packed, x, monkeypatch)
    assert not cold, "a warm route built its projection again"

    plan = vars(layer)[linear._KEPT].dense
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(plan.proof.fail).result()
    cold.clear()
    _compare(layer, packed, x, monkeypatch, expected=serial)
    assert cold, "a retired route bypassed its projection"


@pytest.mark.parametrize("mode", ["stock", "plugin"])
def test_two_threads_run_one_face_for_two_families(mode, build_layer):
    k, n, rows = 2560, 160, 5
    qwen, packed, key = _equivalent_layer(mode, build_layer, n, k)
    other, _, _ = _equivalent_layer(mode, build_layer, n, k, family="llama")
    half = mx.random.normal((rows, k // 2), key=key).astype(mx.float16)
    x = mx.concatenate([half, -half], axis=1)
    # Every layer takes the schedule's route (MLX for a stock bucket, the
    # format's wide tile for a plugin's): the two families share the face's
    # compiled graph, not two arithmetics.
    want = mx.quantized_matmul(x, *packed, transpose=True, bits=8, group_size=64)
    mx.eval(want, x, qwen.parameters(), other.parameters())
    want = np.asarray(want.view(mx.uint16))
    barrier = Barrier(2, timeout=20)

    def run(_unused):
        barrier.wait()
        results = (qwen(x).view(mx.uint16), other(x).view(mx.uint16))
        mx.eval(*results)
        return results

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, (None, None)))
    for qwen_result, other_result in results:
        np.testing.assert_array_equal(np.asarray(other_result), want)
        np.testing.assert_array_equal(np.asarray(qwen_result), want)


def test_a_dot_only_layer_decodes_for_both_families(build_layer, monkeypatch):
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
    # Above MAX_ROWS a format with no tile face decodes once, qwen4-built or
    # another family; the deleted row-exact chunking left no serial half.
    for layer in (qwen, other):
        calls.clear()
        got = np.asarray(layer(x).view(mx.uint16))
        np.testing.assert_array_equal(got, composed)
        assert calls == [id(layer)]


def _gapped_rows(rows, k, key, *, matrices=2):
    half = mx.random.normal((2 * matrices, rows, k // 2), key=key).astype(mx.float16)
    return mx.concatenate([half, -half], axis=-1)[::2]


@pytest.mark.parametrize("mode", ["affine_plugin", "plugin"])
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


@pytest.mark.parametrize("mode", ["plugin"])
def test_batched_format_calls_fold_to_the_dense_route(mode, build_layer, monkeypatch):
    rows, k, n = 5, 2560, 48
    layer, packed, key = _equivalent_layer(mode, build_layer, n, k)
    gapped = _gapped_rows(rows, k, key)
    dense = mx.contiguous(gapped)
    got = _compare(layer, packed, dense, monkeypatch)
    _compare(layer, packed, gapped, monkeypatch, expected=got)
    _compare(layer, packed, dense, monkeypatch)


@pytest.mark.parametrize("mode", ["plugin"])
def test_native_and_python_fallback_wide_dispatch_match_bitwise(
    mode, build_layer, monkeypatch, caplog
):
    from omlx.custom_kernels.quant_matmul import fast as native
    from omlx.formats import linear, tile_dense

    k, n = 2560, 160
    layer, packed, key = _equivalent_layer(mode, build_layer, n, k)
    half = mx.random.normal((2, k // 2), key=key).astype(mx.float16)
    x = mx.concatenate([half, -half], axis=-1)
    cold = _compare(layer, packed, x, monkeypatch)
    if native.has_symbol("FormatLinearFast"):
        fast = layer._fast
        assert fast is not None
        served = []

        def counting(module, value, _fast=fast, _served=served):
            y = _fast(module, value)
            _served.append(y is not None)
            return y

        def missed_native(*_args, **_kwargs):
            pytest.fail("native warm call fell through to the Python lookup")

        layer.__dict__["_fast"] = counting
        try:
            with monkeypatch.context() as patch:
                patch.setattr(tile_dense, "project", missed_native)
                warm = _compare(layer, packed, x, monkeypatch, expected=cold)
        finally:
            layer.__dict__["_fast"] = fast
        assert served == [True], served
    else:
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
    from omlx.formats import tile_dense

    register = FormatLinear._register_fast
    refs = []

    def observe(self, kept, plan, signature, **kwargs):
        graph = plan.binding.state["_kept"].graphs[signature]

        def replay(*args):
            return graph(*args)

        refs.append(weakref.ref(replay))
        plan.proof.keep(signature, replay)
        tile_dense._keep_local(plan, signature, replay)
        register(self, kept, plan, signature, **kwargs)

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
    _compare(
        layer, packed, dense, monkeypatch, expected=folded, mlx_route=mode == "stock"
    )
    assert layer._fast is not None, "folded call registered no native replay"

    broadcast = mx.broadcast_to(gapped[:1], gapped.shape)
    column = mx.repeat(dense, 2, axis=-1)[..., ::2]
    if mode == "stock":
        # The MLX route passes each request's layout to MLX's own qmm, which
        # answers it with its own bits (no fold in this route).
        layouts = [(dense, folded)]
        layouts += [(x, _stock(x, packed, 8)) for x in (gapped, broadcast, column)]
    else:
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
        _compare(
            layer,
            packed,
            x,
            monkeypatch,
            expected=folded[:, 0],
            mlx_route=mode == "stock",
        )
        native_y = layer._fast(layer, x)
        assert native_y is not None, "warm call kept a different plan's dispatcher"
        np.testing.assert_array_equal(
            native_y.view(mx.uint16), folded[:, 0].view(mx.uint16)
        )
    separate = mx.stack([layer(row) for row in dense])
    assert not mx.array_equal(folded.view(mx.uint16), separate.view(mx.uint16)).item()


def test_planned_plugin_3d_5row_does_not_take_stock_matmul(build_layer, monkeypatch):
    """Only stock layers divert to _stock_matmul on losing decode buckets; a
    planned plugin layer stays on its format route."""
    layer, packed, key = _equivalent_layer("plugin", build_layer, 160, 2560)
    assert not layer._stock
    x = mx.random.normal((5, 1, 2560), key=key).astype(mx.float16)
    _compare(layer, packed, x, monkeypatch)


@pytest.mark.parametrize("change", ["replace", "union", "placeholder"])
def test_writer_mutation_retires_native_replay(change, build_layer, tmp_path):
    # Rows 33 take the unmeasured >=17 default dense route (a <17 row now
    # takes the lead-H4 measured-DATA-or-MLX route, which keeps no native
    # replay); the retirement is judged against the layer's own output.
    layer, packed, key = _projection(build_layer, 160, 2560, 8, mx.float16)
    x = mx.random.normal((33, 1, 2560), key=key).astype(mx.float16)
    before = layer(x).view(mx.uint16)
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
        after = cloned(x).view(mx.uint16)
        assert not mx.array_equal(
            before, after
        ).item(), "the clone must recompute with the mutated scales"
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
    tag = "format_linear_tile_nax" if linear._nax_host() else "format_linear_tile_steel"
    assert any(
        key[0] == tag and face in key and proof.ran
        for key, proof in kernels._PROOFS.items()
    )


@pytest.mark.parametrize("arch", ["unknown", "17s", "applegpu_gXs"])
def test_unrecognized_metal_architecture_uses_the_dot_route(
    arch, build_layer, monkeypatch
):
    from omlx.formats import linear, tile_dense, tuning

    qwen, _, key = _equivalent_layer("plugin", build_layer, 16, 2560)
    other, _, _ = _equivalent_layer("plugin", build_layer, 16, 2560, family="llama")
    monkeypatch.setattr(tuning, "current_chip", lambda: f"{arch}_c40")

    def forbidden(*args):
        pytest.fail("an unknown chip entered the tile schedule")

    monkeypatch.setattr(tile_dense, "project", forbidden)
    x = mx.random.normal((2, 2560), key=key).astype(mx.float16)
    np.testing.assert_array_equal(qwen(x).view(mx.uint16), other(x).view(mx.uint16))
    kept = vars(qwen)[linear._KEPT]
    assert kept.dense not in (None, linear._UNBUILT)
    assert tile_dense.route_for(kept.dense, 2, x.dtype).backend == "dot"


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
            ("EXACT_ROWS", 0),
        ],
        grid=(32, 4, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(16,)],
        output_dtypes=[mx.float16],
        init_value=-31,
    )
    np.testing.assert_array_equal(np.asarray(got), [k] * n + [-31] * (16 - n))


def test_planless_stock_affine_short_calls_keep_mlx_qmv(build_layer, monkeypatch):
    # K 128 at 8 bits is MLX's qmv_quad: the affine format has no dot face,
    # so the layer has no plan. A short call keeps MLX's own matmul, a row's
    # bits each, and never decodes the matrix.
    for dtype in (mx.bfloat16, mx.float16):
        w = mx.random.normal((256, 128), key=mx.random.key(211)).astype(dtype)
        packed = mx.quantize(w, group_size=128, bits=8)
        names = ("weight", "scales", "biases")
        layer = build_layer("affine", 8, 128, 128, dict(zip(names, packed)))
        assert linear._plan(layer) is None

        def trap(self, *_args, **_kwargs):
            raise AssertionError("a plan-less stock layer decoded its matrix")

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


def test_a_plugin_long_call_takes_the_scheduled_dense_route(build_layer):
    # 513 rows at N 512 / K 2560: the construction-time schedule serves the
    # plugin face's dense tile; no Dot rows route (the deleted ordinary rows
    # kernel) and no Dot replay are kept for it.
    from omlx.formats import tile_dense

    layer, _, _ = _affine_plugin(build_layer, 512, 2560, 4, mx.float16)
    got = layer(mx.zeros((513, 2560), mx.float16))
    mx.eval(got)
    kept = vars(layer)[linear._KEPT]
    assert kept.dense not in (None, linear._UNBUILT)
    assert tile_dense.route_for(kept.dense, 513, mx.float16).backend in ("nax", "steel")
    assert kept.dot_fast is None


def test_an_mma_route_registers_its_own_native_replay(build_layer, monkeypatch):
    """A measured mma bucket runs the verify skeleton and its compiled graph
    shares the single 3-operand dense_fast replay (inlining N_size inside
    the compiled graph so mma/msg never miss a dense_fast slot warmed by
    mlx/wide/nax/steel; lead 2b ruling: every route at 2..16 through the
    native fast)."""
    from omlx.custom_kernels.quant_matmul import fast as native
    from omlx.formats import tile_dense, tuning

    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires the native warm dispatcher")
    n, k, rows = 128, 512, 8
    layer, packed, key = _projection(build_layer, n, k, 4, mx.bfloat16)
    policy = tuning.MmaTilePolicy(bn=16, bk=32, nsg=16)
    monkeypatch.setattr(
        tuning,
        "lookup",
        lambda route, *a, **kw: policy if route == "mma_tile" else None,
    )
    x = mx.random.normal((rows, k), key=key).astype(mx.bfloat16)
    first = layer(x)
    mx.eval(first)
    kept = vars(layer)[linear._KEPT]
    assert tile_dense.route_for(kept.dense, rows, x.dtype).backend == "mma"
    assert not hasattr(kept, "dense_fast_nsize")
    fast = kept.dense_fast
    assert fast is not None, "the mma route registered no dense_fast replay"
    served = []

    def counting(module, value, _fast=fast, _served=served):
        y = _fast(module, value)
        _served.append(y is not None)
        return y

    with monkeypatch.context() as patch:
        layer.__dict__["_fast"] = counting
        patch.setattr(
            tile_dense,
            "project",
            lambda *a, **kwargs: pytest.fail("a warm mma call projected"),
        )
        try:
            again = layer(x)
            mx.eval(again)
        finally:
            layer.__dict__["_fast"] = fast
    assert served == [True], served
    np.testing.assert_array_equal(
        np.asarray(first.view(mx.uint16)), np.asarray(again.view(mx.uint16))
    )
    stock = _stock(x, packed, 4)
    mx.eval(stock)
    weight = mx.dequantize(
        packed[0], packed[1], packed[2], group_size=64, bits=4
    ).astype(mx.float32)
    reference = x.astype(mx.float32) @ weight.T

    def rel(array):
        return float(
            mx.sqrt(mx.sum((array.astype(mx.float32) - reference) ** 2))
            / mx.sqrt(mx.sum(reference**2))
        )

    assert rel(first) <= 1.05 * rel(stock) + 1e-6, (rel(first), rel(stock))


def test_concurrent_first_calls_share_one_dense_plan_and_fast(build_layer, monkeypatch):
    """Concurrent first calls on a fresh FormatLinear bind one dense plan and
    one native replay without a hot-path lock, so later signatures still hit
    ``layer._fast`` (review MEDIUM-1, rule 28)."""
    from omlx.custom_kernels.quant_matmul import fast as native
    from omlx.formats import tile_dense

    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires the native warm dispatcher")
    k, n = 2560, 160
    layer, _, key = _projection(build_layer, n, k, 8, mx.bfloat16)
    x = mx.random.normal((4, k), key=key).astype(mx.bfloat16)
    x2 = mx.random.normal((6, k), key=mx.random.key(99)).astype(mx.bfloat16)
    mx.eval(x, x2, layer.parameters())

    real_build = tile_dense.build_plan
    barrier = Barrier(2, timeout=15)
    first_done = threading.Event()
    winner_thread = []
    winner_lock = threading.Lock()
    seen_dense = []

    def racing_build(held, **kwargs):
        plan = real_build(held, **kwargs)
        with winner_lock:
            is_first = not winner_thread
            if is_first:
                winner_thread.append(threading.get_ident())
        barrier.wait()
        if not is_first:
            assert first_done.wait(timeout=15), "first thread did not finish call"
        return plan

    monkeypatch.setattr(tile_dense, "build_plan", racing_build)

    def call_once(_unused):
        y = layer(x)
        mx.eval(y)
        if winner_thread and threading.get_ident() == winner_thread[0]:
            first_done.set()
        seen_dense.append(vars(layer)[linear._KEPT].dense)
        return np.asarray(y.view(mx.uint16))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(call_once, range(2)))

    np.testing.assert_array_equal(results[0], results[1])
    kept = vars(layer)[linear._KEPT]
    assert kept.dense is not None
    assert seen_dense[0] is kept.dense and seen_dense[1] is kept.dense
    assert layer._fast is not None and layer._fast is kept.dense_fast

    # A new shape on the main thread must compile into the same binding that
    # layer._fast reads and hit native replay immediately after.
    y2 = layer(x2)
    mx.eval(y2)
    replay = layer._fast(layer, x2)
    assert replay is not None, "layer._fast is decoupled from kept.dense.binding"
    mx.eval(replay)
    np.testing.assert_array_equal(
        np.asarray(y2.view(mx.uint16)), np.asarray(replay.view(mx.uint16))
    )

    # Same invariant when two threads race inside _keep() itself (before
    # _dense_tile): the winning _Kept record stays installed and coupled to
    # layer._fast so later shapes still hit native replay.
    fresh, _, _ = _projection(build_layer, n, k, 8, mx.bfloat16)
    mx.eval(fresh.parameters())
    real_plan = linear._plan
    keep_barrier = Barrier(2, timeout=15)
    keep_done = threading.Event()
    keep_winner = []
    keep_lock = threading.Lock()
    seen_kept = []

    def racing_plan(held):
        plan = real_plan(held)
        with keep_lock:
            is_first = not keep_winner
            if is_first:
                keep_winner.append(threading.get_ident())
        keep_barrier.wait()
        if not is_first:
            assert keep_done.wait(timeout=15), "first thread did not finish call"
        return plan

    monkeypatch.setattr(tile_dense, "build_plan", real_build)
    monkeypatch.setattr(linear, "_plan", racing_plan)

    def call_fresh(_unused):
        y = fresh(x)
        mx.eval(y)
        if keep_winner and threading.get_ident() == keep_winner[0]:
            keep_done.set()
        seen_kept.append(vars(fresh)[linear._KEPT])
        return np.asarray(y.view(mx.uint16))

    with ThreadPoolExecutor(max_workers=2) as pool:
        fresh_results = list(pool.map(call_fresh, range(2)))

    np.testing.assert_array_equal(fresh_results[0], fresh_results[1])
    kept_fresh = vars(fresh)[linear._KEPT]
    assert seen_kept[0] is kept_fresh and seen_kept[1] is kept_fresh
    assert fresh._fast is not None and fresh._fast is kept_fresh.dense_fast
    mx.eval(fresh(x2))
    assert fresh._fast(fresh, x2) is not None


def test_publish_fast_recovers_orphaned_fast2_without_duplicating_slots(
    build_layer, monkeypatch
):
    """A warm route whose ``_fast`` was orphaned while its replay sat in
    ``_fast2`` re-publishes it (and drops the duplicate), and a second live
    route still binds ``_fast2`` afterwards (review LOW on f29d0c98e)."""
    from omlx.custom_kernels.quant_matmul import fast as native
    from omlx.formats import tuning

    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires the native warm dispatcher")
    k, n = 2560, 160
    layer, _, key = _projection(build_layer, n, k, 8, mx.bfloat16)
    # The routes of an unmeasured chip: rows 2-2 (mlx) and 4-4 (dot).
    monkeypatch.setattr(tuning, "row_bounds", lambda route, **kwargs: (2, 4, 16))

    def lookup(route, *args, **kwargs):
        if route == "dense_tile" and kwargs.get("rows") == 4:
            return tuning.DenseTilePolicy("dot")
        return None

    monkeypatch.setattr(tuning, "lookup", lookup)
    x2 = mx.random.normal((2, k), key=key).astype(mx.bfloat16)
    x4 = mx.random.normal((4, k), key=key).astype(mx.bfloat16)
    mx.eval(layer(x2), layer(x4))
    assert layer._fast is not None

    layer["weight"] = mx.array(layer["weight"])
    mx.eval(layer(x2))
    kept = vars(layer)[linear._KEPT]
    new_fast = kept.dense_fast
    assert new_fast is not None
    # The post-retraction interleaving: _fast popped, the live replay left in
    # _fast2.
    layer.__dict__.pop(linear._FAST, None)
    layer.__dict__[linear._FAST2] = new_fast
    mx.eval(layer(x2))
    assert layer._fast is new_fast
    assert layer._fast2 is None

    mx.eval(layer(x4))
    assert layer._fast2 is kept.dot_fast


def test_warm_secondary_route_recovers_cleared_fast2(build_layer, monkeypatch):
    """A warm route whose replay lost ``_fast2`` (while the other replay holds
    ``_fast``) re-publishes it, a dot replay never duplicates itself across
    ``_fast``/``_fast2``, and publishing the replay already in ``_fast`` keeps
    a distinct ``_fast2`` (review LOW-1 on 7cfba7d15)."""
    from omlx.custom_kernels.quant_matmul import fast as native
    from omlx.formats import tuning

    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires the native warm dispatcher")
    k, n = 2560, 160
    layer, _, key = _projection(build_layer, n, k, 8, mx.bfloat16)
    monkeypatch.setattr(tuning, "row_bounds", lambda route, **kwargs: (2, 4, 16))

    def lookup(route, *args, **kwargs):
        if route == "dense_tile" and kwargs.get("rows") == 4:
            return tuning.DenseTilePolicy("dot")
        return None

    monkeypatch.setattr(tuning, "lookup", lookup)
    x2 = mx.random.normal((2, k), key=key).astype(mx.bfloat16)
    x4 = mx.random.normal((4, k), key=key).astype(mx.bfloat16)
    mx.eval(layer(x2), layer(x4))
    kept = vars(layer)[linear._KEPT]
    dense_fast, dot_fast = kept.dense_fast, kept.dot_fast
    assert dense_fast is not None and dot_fast is not None

    # _fast holds the dense replay, _fast2 is empty: a warm dot call
    # re-publishes the dot replay.
    layer.__dict__[linear._FAST] = dense_fast
    layer.__dict__.pop(linear._FAST2, None)
    mx.eval(layer(x4))
    assert layer._fast2 is dot_fast

    # Symmetric: _fast holds the dot replay, a warm dense call re-publishes.
    layer.__dict__[linear._FAST] = dot_fast
    layer.__dict__.pop(linear._FAST2, None)
    mx.eval(layer(x2))
    assert layer._fast2 is dense_fast

    # An orphaned dot replay in _fast2 with no _fast lands alone in _fast.
    layer.__dict__.pop(linear._FAST, None)
    layer.__dict__[linear._FAST2] = dot_fast
    mx.eval(layer(x4))
    assert layer._fast is dot_fast and layer._fast2 is None

    # Publishing the replay already in _fast keeps the distinct _fast2.
    layer.__dict__[linear._FAST] = dense_fast
    layer.__dict__[linear._FAST2] = dot_fast
    FormatLinear._publish_fast(layer, kept, dense_fast)
    assert layer._fast is dense_fast and layer._fast2 is dot_fast


def test_publish_fast_else_branch_clears_duplicate_slot():
    """The setdefault branch of ``_publish_fast`` must not leave the replay
    it won ``_fast`` with duplicated in ``_fast2`` (review LOW-2)."""

    class _RacingStore(dict):
        def __init__(self, initial, swap):
            super().__init__(initial)
            self.swap = swap
            self.swapped = False

        def setdefault(self, key, default=None):
            if key == linear._FAST and not self.swapped:
                self.swapped = True
                old = dict.get(self, key)
                dict.__setitem__(self, key, self.swap)
                return old
            return dict.setdefault(self, key, default)

    class _Proxy:
        def __init__(self, store):
            object.__setattr__(self, "_store", store)

        @property
        def __dict__(self):
            return self._store

    old_fast, new_fast = object(), object()
    kept = object()
    store = _RacingStore({linear._KEPT: kept, linear._FAST: old_fast}, new_fast)
    FormatLinear._publish_fast(_Proxy(store), kept, new_fast)
    assert store.get(linear._FAST) is new_fast
    assert linear._FAST2 not in store


def test_dot_only_layer_recovers_orphaned_fast2(build_layer, monkeypatch):
    """A dot-only layer's warm plan-graph replay re-publishes its native
    replay when ``_fast`` was orphaned (review LOW-3)."""
    from omlx.custom_kernels.quant_matmul import fast as native

    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires the native warm dispatcher")
    k, n = 2560, 160
    layer, _, key = _equivalent_layer("dot_only", build_layer, n, k)
    x2 = mx.random.normal((2, k), key=key).astype(mx.float16)
    mx.eval(layer(x2))
    kept = vars(layer)[linear._KEPT]
    dot_fast = kept.dot_fast
    assert dot_fast is not None

    layer.__dict__.pop(linear._FAST, None)
    layer.__dict__[linear._FAST2] = dot_fast
    mx.eval(layer(x2))
    assert layer._fast is dot_fast and layer._fast2 is None


def test_concurrent_native_fast_creation_shares_one_dispatcher(
    build_layer, monkeypatch
):
    """Two threads racing to create the same native replay slot keep a single
    FormatLinearFast instance via atomic publication, so both signatures hit
    the published ``kept.dense_fast`` replay."""
    from omlx.custom_kernels.quant_matmul import fast as native

    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires the native warm dispatcher")
    k, n = 2560, 160
    layer, _, key = _projection(build_layer, n, k, 8, mx.bfloat16)
    # Build kept and its dense plan first so both threads share the exact same
    # dense plan and race specifically on creating kept.dense_fast.
    x1 = mx.random.normal((1, k), key=key).astype(mx.bfloat16)
    x4 = mx.random.normal((4, k), key=mx.random.key(101)).astype(mx.bfloat16)
    x6 = mx.random.normal((6, k), key=mx.random.key(102)).astype(mx.bfloat16)
    mx.eval(x1, x4, x6, layer.parameters())
    mx.eval(layer(x1))
    kept = vars(layer)[linear._KEPT]
    assert layer._dense_tile(kept) is not None and kept.dense_fast is None

    real_fast = native.format_linear_fast
    barrier = Barrier(2, timeout=15)

    def racing_fast(*args, **kwargs):
        built = real_fast(*args, **kwargs)
        barrier.wait()
        return built

    monkeypatch.setattr(native, "format_linear_fast", racing_fast)

    def run(arr):
        y = layer(arr)
        mx.eval(y)
        return y

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, (x4, x6)))

    assert kept.dense_fast is not None and kept.dense_fast in (
        layer._fast,
        layer._fast2,
    )
    # Replay both signatures on the main thread after warming its thread-local
    # graph cache.
    for arr in (x4, x6):
        mx.eval(layer(arr))
        assert kept.dense_fast(layer, arr) is not None, arr.shape


def test_a_layer_serves_both_fast_groups_through_their_own_replays(
    build_layer, monkeypatch
):
    """A layer whose schedule mixes an mma bucket and a Dot bucket keeps one
    native replay per operand set; a warm call whose signature is in the
    *other* replay is still served by its own (never a Python projection or
    the Dot kernel's _launch)."""
    from omlx.custom_kernels.quant_matmul import fast as native
    from omlx.formats import tile_dense, tuning

    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires the native warm dispatcher")
    n, k = 128, 512
    layer, packed, key = _equivalent_layer("plugin", build_layer, n, k)
    monkeypatch.setattr(tile_dense, "_wide_legal", lambda face, k, chip: False)
    pol = tuning.MmaTilePolicy(bn=16, bk=32, nsg=16)

    def lookup(route, *args, **kwargs):
        if route == "mma_tile" and kwargs.get("rows") == 8:
            return pol
        return None

    monkeypatch.setattr(tuning, "lookup", lookup)
    x4 = mx.random.normal((4, k), key=key).astype(mx.bfloat16)
    x8 = mx.random.normal((8, k), key=key).astype(mx.bfloat16)
    first4, first8 = layer(x4), layer(x8)
    mx.eval(first4, first8)
    kept = vars(layer)[linear._KEPT]
    assert tile_dense.route_for(kept.dense, 4, x4.dtype).backend == "dot"
    assert tile_dense.route_for(kept.dense, 8, x8.dtype).backend == "mma"
    mma_fast, dot_fast = kept.dense_fast, kept.dot_fast
    assert mma_fast is not None and dot_fast is not None

    def forbidden_project(*args, **kwargs):
        pytest.fail("a warm call projected instead of its own replay")

    def forbidden_launch(*args, **kwargs):
        pytest.fail("a warm dot call launched the Dot kernel again")

    # The mma signature with the Dot replay on top of the layer (and _fast2
    # cleared so the secondary slot lookup inside _scheduled is exercised).
    orig_fast2 = layer.__dict__.get("_fast2")
    with monkeypatch.context() as patch:
        layer.__dict__["_fast"] = dot_fast
        layer.__dict__.pop("_fast2", None)
        patch.setattr(tile_dense, "project", forbidden_project)
        try:
            again8 = layer(x8)
            mx.eval(again8)
        finally:
            layer.__dict__["_fast"] = mma_fast
            if orig_fast2 is not None:
                layer.__dict__["_fast2"] = orig_fast2
    np.testing.assert_array_equal(
        np.asarray(first8.view(mx.uint16)), np.asarray(again8.view(mx.uint16))
    )
    # The Dot signature with the mma replay on top of the layer.
    with monkeypatch.context() as patch:
        layer.__dict__["_fast"] = mma_fast
        patch.setattr(linear, "_launch", forbidden_launch)
        try:
            again4 = layer(x4)
            mx.eval(again4)
        finally:
            layer.__dict__["_fast"] = dot_fast
    np.testing.assert_array_equal(
        np.asarray(first4.view(mx.uint16)), np.asarray(again4.view(mx.uint16))
    )


def test_a_plugin_rows_call_is_served_by_its_native_replay(build_layer, monkeypatch):
    from omlx.custom_kernels.quant_matmul import fast as native
    from omlx.formats import tile_dense

    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires the native warm dispatcher")
    layer, _, _ = _affine_plugin(build_layer, 160, 2560, 4, mx.float16)
    x = mx.random.normal((4, 2560)).astype(mx.float16)
    first = layer(x)
    mx.eval(first)
    fast = layer._fast
    assert fast is not None, "the plugin rows call registered no native replay"
    served = []

    def counting(module, value, _fast=fast, _served=served):
        y = _fast(module, value)
        _served.append(y is not None)
        return y

    with monkeypatch.context() as patch:
        layer.__dict__["_fast"] = counting
        patch.setattr(
            tile_dense,
            "project",
            lambda *a, **k: pytest.fail("a warm plugin rows call projected"),
        )
        try:
            again = layer(x)
            mx.eval(again)
        finally:
            layer.__dict__["_fast"] = fast
    assert served == [True], served
    np.testing.assert_array_equal(
        np.asarray(first.view(mx.uint16)), np.asarray(again.view(mx.uint16))
    )


def test_planless_stock_affine_empty_input_avoids_decode(build_layer, monkeypatch):
    w = mx.random.normal((256, 128), key=mx.random.key(311)).astype(mx.float16)
    packed = mx.quantize(w, group_size=128, bits=8)
    names = ("weight", "scales", "biases")
    layer = build_layer("affine", 8, 128, 128, dict(zip(names, packed)))

    def trap(self, *_args, **_kwargs):
        raise AssertionError("an empty plan-less stock call decoded its matrix")

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


def test_planless_stock_affine_above_max_rows_takes_the_dense_tile(
    build_layer, monkeypatch
):
    # A plan-less qmv_quad stock layer keeps MLX's own matmul up to MAX_ROWS
    # (16) and above it takes the format's dense tile; the deleted row-exact
    # chunking left no serial route, and neither call decodes.
    monkeypatch.setattr(
        FormatLinear,
        "_decoded",
        lambda self, *a, **k: pytest.fail("a plan-less stock call decoded"),
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

        x = mx.random.normal((4, 128), key=mx.random.key(19)).astype(dtype)
        stock = mx.quantized_matmul(x, *packed, transpose=True, bits=8, group_size=128)
        calls.clear()
        y = layer(x)
        assert y.shape == (4, 256)
        assert mx.array_equal(y.view(mx.uint16), stock.view(mx.uint16)).item()

        x = mx.random.normal((17, 128), key=mx.random.key(19)).astype(dtype)
        calls.clear()
        y = layer(x)
        assert y.shape == (17, 256)
        assert mx.array_equal(
            y.view(mx.uint16), _one_expert_of(packed, 8, 128, x).view(mx.uint16)
        ).item()
        assert calls == [], calls


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

    monkeypatch.setattr(
        FormatLinear,
        "_decoded",
        lambda self, *a, **k: pytest.fail("short plan-less stock call decoded"),
    )
    for rows_count in (12, linear.MAX_ROWS):
        x = mx.random.normal((rows_count, 128), key=mx.random.key(23)).astype(
            mx.bfloat16
        )
        want = mx.quantized_matmul(x, *packed, transpose=True, bits=8, group_size=128)
        y = layer(x)
        assert y.shape == (rows_count, 256)
        assert mx.array_equal(y.view(mx.uint16), want.view(mx.uint16)).item()


def test_planless_plugin_skips_stock_matmul(build_layer, monkeypatch):
    # A plugin format with no dot face never calls _stock_matmul, and rows ==
    # MAX_ROWS stays on the non-prefill path.
    layer, _, _ = _affine_plugin(build_layer, 128, 64, 4, mx.bfloat16)
    assert linear._plan(layer) is None
    calls = []
    orig_stock = FormatLinear._stock_matmul

    def counted(self, x):
        calls.append(x.shape)
        return orig_stock(self, x)

    monkeypatch.setattr(FormatLinear, "_stock_matmul", counted)
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


def test_planless_plugin_above_max_rows_takes_the_dense_tile(build_layer, monkeypatch):
    """A dot-less plugin above MAX_ROWS takes the format's dense tile, the
    rule-29 survivor of the deleted row-exact serial path; it never decodes."""

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
    half = mx.random.normal((rows, k // 2), key=keys[1]).astype(mx.bfloat16)
    x = mx.concatenate([half, -half], axis=1)
    serial = mx.concatenate([layer(x[r : r + 1]) for r in range(rows)], axis=0)
    tile = _one_expert_of(packed, 4, 64, x)
    assert not mx.array_equal(
        tile.view(mx.uint16), serial.view(mx.uint16)
    ).item(), "fixture must distinguish the dense tile from a 1-row serial call"
    monkeypatch.setattr(
        FormatLinear,
        "_decoded",
        lambda *arguments, **keywords: pytest.fail(
            "a dot-less plugin call decoded its matrix"
        ),
    )
    y = layer(x)
    assert mx.array_equal(y.view(mx.uint16), tile.view(mx.uint16)).item()


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


def _c2_ok(got, stock, x, packed, bits, group_size=64):
    weight = mx.dequantize(
        packed[0], packed[1], packed[2], group_size=group_size, bits=bits
    ).astype(mx.float32)
    rows = x.size // x.shape[-1]
    reference = (x.astype(mx.float32).reshape(rows, -1) @ weight.T).reshape(
        *x.shape[:-1], packed[0].shape[0]
    )

    def rel(array):
        return float(
            mx.sqrt(mx.sum((array.astype(mx.float32) - reference) ** 2))
            / mx.sqrt(mx.sum(reference**2))
        )

    got_err, stock_err = rel(got), rel(stock)
    assert got_err <= 1.05 * stock_err + 1e-6, (got_err, stock_err)


def test_mma_and_msg_shared_proof_isolate_distinct_output_widths(
    build_layer, monkeypatch
):
    """Two layers sharing (route, face, dtypes, k) with different N must not
    replay each other's compiled mma/msg graph (plan.n is part of shared_key)."""
    from omlx.formats import tile_dense, tuning

    k = 2560
    mma_pol = tuning.MmaTilePolicy(bn=16, bk=32, nsg=8)
    monkeypatch.setattr(
        tuning,
        "lookup",
        lambda route, *a, **kw: mma_pol if route == "mma_tile" else None,
    )
    small, p_small, key = _projection(build_layer, 512, k, 4, mx.bfloat16)
    wide, p_wide, _ = _projection(build_layer, 10240, k, 4, mx.bfloat16)
    x = mx.random.normal((8, k), key=key).astype(mx.bfloat16)
    y_small = small(x)
    y_wide = wide(x)
    mx.eval(y_small, y_wide)
    assert y_small.shape == (8, 512)
    assert y_wide.shape == (8, 10240)
    assert (
        tile_dense.route_for(vars(wide)[linear._KEPT].dense, 8, x.dtype).backend
        == "mma"
    )
    _c2_ok(y_small, _stock(x, p_small, 4), x, p_small, 4)
    _c2_ok(y_wide, _stock(x, p_wide, 4), x, p_wide, 4)

    msg_pol = tuning.MsgTilePolicy(nsg=8)
    monkeypatch.setattr(
        tuning,
        "lookup",
        lambda route, *a, **kw: msg_pol if route == "msg_tile" else None,
    )
    m_small, mp_small, key2 = _projection(build_layer, 256, k, 8, mx.bfloat16)
    m_wide, mp_wide, _ = _projection(build_layer, 512, k, 8, mx.bfloat16)
    x6 = mx.random.normal((6, k), key=key2).astype(mx.bfloat16)
    ym_small = m_small(x6)
    ym_wide = m_wide(x6)
    mx.eval(ym_small, ym_wide)
    assert ym_small.shape == (6, 256)
    assert ym_wide.shape == (6, 512)
    assert (
        tile_dense.route_for(vars(m_wide)[linear._KEPT].dense, 6, x6.dtype).backend
        == "msg"
    )
    _c2_ok(ym_small, _serial_stock(x6, mp_small, 8), x6, mp_small, 8)
    _c2_ok(ym_wide, _serial_stock(x6, mp_wide, 8), x6, mp_wide, 8)


def test_mma_and_msg_stand_in_never_reads_live_layer_tensors(build_layer, monkeypatch):
    """Proof.prove's stand-in must supply fresh zeros/N_size, never live layer
    operands or plan.n_size (kernels.py Proof.prove contract)."""
    from omlx.formats import tuning

    seen = []
    real_prove = kernels.Proof.prove

    def capture(self, metal, build, *args, **kwargs):
        if not self.failed and metal not in self.ran:
            inputs, fn = build()
            seen.append((metal, tuple(inputs)))
            return real_prove(self, metal, lambda: (inputs, fn), *args, **kwargs)
        return real_prove(self, metal, build, *args, **kwargs)

    monkeypatch.setattr(kernels.Proof, "prove", capture)
    for route_name, pol, rows, bits in (
        ("mma_tile", tuning.MmaTilePolicy(16, 32, 8), 8, 4),
        ("msg_tile", tuning.MsgTilePolicy(8), 4, 8),
    ):
        seen.clear()
        monkeypatch.setattr(
            tuning,
            "lookup",
            lambda r, *a, _rn=route_name, _p=pol, **kw: _p if r == _rn else None,
        )
        layer, _, key = _projection(build_layer, 256, 512, bits, mx.bfloat16)
        x = mx.random.normal((rows, 512), key=key).astype(mx.bfloat16)
        with testing.fresh_proofs():
            mx.eval(layer(x))
            plan = vars(layer)[linear._KEPT].dense
            assert len(seen) == 1, (route_name, seen)
            live = {id(a) for a in plan.operands}
            for arr in seen[0][1]:
                assert id(arr) not in live, f"{route_name} stand-in read live tensor"
            # A second call on another shape of the same Metal instantiation
            # reuses Proof.ran without re-proving.
            x3d = x.reshape(1, rows, 512)
            mx.eval(layer(x3d))
            assert len(seen) == 1, f"{route_name} re-proved on same Metal"


def test_cross_route_specificity_lets_exact_mma_beat_coarse_dense_tile(
    tmp_path, build_layer, monkeypatch
):
    """An exact-row mma_tile/msg_tile entry must beat a coarser dense_tile
    (<17 or wildcard) entry in the same chip table."""
    from omlx.formats import affine, tile_dense, tuning, verify_skeleton

    verify_skeleton._register_tuning()
    chip = "applegpu_g17s_c40"
    face = affine.tile_face(4, 64, mx.bfloat16)
    coarse_dense = {
        "key": {
            "route": "dense_tile",
            "face_id": face.key,
            "dtype": "bf16",
            "bits": 4,
            "group": 64,
            "rows_bucket": "<17",
            "n": 512,
            "k": 2560,
        },
        "face_sha256": tuning.face_digest(face),
        "skeleton_sha256": tuning.skeleton_digest("dense_tile"),
        "measured_at": "2026-11-23T23:00:00Z",
        "exactness": "bitwise",
        "params": {"backend": "mlx"},
    }
    exact_mma = {
        "key": {
            "route": "mma_tile",
            "face_id": face.key,
            "dtype": "bf16",
            "bits": 4,
            "group": 64,
            "rows_bucket": "8-8",
            "n": 512,
            "k": 2560,
        },
        "face_sha256": tuning.face_digest(face),
        "skeleton_sha256": tuning.skeleton_digest("mma_tile"),
        "measured_at": "2026-11-23T23:00:00Z",
        "exactness": "rule17_fp32",
        "params": {"bn": 16, "bk": 32, "nsg": 8},
    }
    v1 = tmp_path / "v1"
    v1.mkdir()
    (v1 / f"{chip}.json").write_text(
        json.dumps(
            {"schema_version": 1, "chip": chip, "entries": [coarse_dense, exact_mma]}
        )
    )
    monkeypatch.setattr(tuning, "_TABLE_DIRS", [v1])
    monkeypatch.setattr(tuning, "_EXTRA_DIRS", [])
    tuning.reset_cache()
    try:
        layer, _, _ = _projection(build_layer, 512, 2560, 4, mx.bfloat16)
        plan = tile_dense.build_plan(layer, tensor_units=True, chip=chip)
        assert plan is not None
        assert tile_dense.route_for(plan, 4, mx.bfloat16).backend == "mlx"
        assert tile_dense.route_for(plan, 8, mx.bfloat16).backend == "mma"
    finally:
        tuning.reset_cache()


def test_schedule_construction_invariant_legality_and_zero_fast_thrash(
    build_layer, monkeypatch
):
    """Commit 3 Tests #1, #3, #4 & review-3 F1(c):
    - zero tuning.lookup after plan construction across warm & cold M=1..24;
    - narrow/odd shapes (N=1,4,48; K=320) & K%32!=0,K%8==0 legality fail-closed;
    - alternating (1,1,K) / (1,8,K) mma / (1,33,K) dense never overwrites _fast."""
    from omlx.custom_kernels.quant_matmul import fast as native
    from omlx.formats import tile_dense, tuning

    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires the native warm dispatcher")

    # Zero tuning.lookup after plan construction + zero _fast thrashing across
    # rows 1 (Dot), 8/17 (mma, N_size replay) and 33 (dense, plain replay).
    pol_mma = tuning.MmaTilePolicy(bn=16, bk=32, nsg=8)

    def _lookup(route, *args, **kwargs):
        rows = kwargs.get("rows")
        if route == "mma_tile" and rows in (8, 17, 24):
            return pol_mma
        return None

    monkeypatch.setattr(tuning, "lookup", _lookup)
    layer, packed, key = _projection(build_layer, 128, 512, 4, mx.bfloat16)
    x33 = mx.random.normal((1, 33, 512), key=key).astype(mx.bfloat16)
    mx.eval(layer(x33))
    first_fast = layer._fast
    assert first_fast is not None

    monkeypatch.setattr(
        tuning,
        "lookup",
        lambda *a, **k: pytest.fail("tuning.lookup called after plan construction"),
    )
    for m in (1, 2, 4, 8, 16, 17, 24, 33, 8, 17, 1, 33):
        xm = mx.random.normal((1, m, 512), key=key).astype(mx.bfloat16)
        ym = layer(xm)
        mx.eval(ym)
        assert ym.shape == (1, m, 128)
        assert layer._fast is first_fast, f"_fast overwritten at m={m}"
        _c2_ok(ym, _stock(xm, packed, 4), xm, packed, 4)

    # Narrow/odd shapes fail closed on mma/msg and fall back cleanly with C2 PASS.
    tuning.reset_cache()
    monkeypatch.undo()
    for n, k in ((1, 2560), (4, 10240), (48, 2560), (10240, 320)):
        narrow, npk, nkey = _projection(build_layer, n, k, 8, mx.bfloat16)
        xn = mx.random.normal((1, 4, k), key=nkey).astype(mx.bfloat16)
        yn = narrow(xn)
        mx.eval(yn)
        assert yn.shape == (1, 4, n)
        _c2_ok(yn, _stock(xn, npk, 8), xn, npk, 8)

    # K % 32 != 0, K % 8 == 0 on a plugin face takes tile_wide at rows=4 instead
    # of hitting a blanket K % 32 bailout.
    from tests.test_formats_kernels import _toy
    from tests.test_formats_tile_prefill import _TILE, _tile_toy, _ToyFormat

    k_tail = 40
    tile_face = _tile_toy("ktail40", _TILE.replace("GROUP = 8", "GROUP = 8"))
    plugin = _ToyFormat(_toy("ktail40"), tile_face)
    codes = mx.random.randint(0, 256, shape=(16, k_tail), key=key).astype(mx.uint8)
    tail_layer = build_layer(
        plugin, 8, 8, k_tail, {"codes": codes, "group_scales": mx.ones((16,))}
    )
    xt = mx.random.normal((4, k_tail), key=key).astype(mx.float16)
    yt = tail_layer(xt)
    mx.eval(yt)
    kept_tail = vars(tail_layer)[linear._KEPT]
    assert tile_dense.route_for(kept_tail.dense, 4, mx.float16).backend == "wide"


def test_strided_3d_native_fast_replay_matches_contiguous(build_layer):
    """Commit 3 Test #5 & review-3 M6: non-vacuous FormatLinearFast replay on
    strided 3-D slices for both DensePlan (4,4,K) and dense_tile (2,17,K)."""
    from omlx.custom_kernels.quant_matmul import fast as native

    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires the native warm dispatcher")
    k, n = 512, 128
    layer, _, key = _projection(build_layer, n, k, 4, mx.bfloat16)
    for shape in ((4, 4, k), (2, 17, k)):
        b, m, _ = shape
        warm = mx.random.normal(shape, key=key).astype(mx.bfloat16)
        mx.eval(layer(warm))
        fast = layer._fast
        assert fast is not None, shape
        big = mx.random.normal((b, m * 2, k), key=key).astype(mx.bfloat16)
        mx.eval(big)
        strided = big[:, ::2, :]
        assert strided.shape == shape
        y_fast = fast(layer, strided)
        assert y_fast is not None, f"native _fast missed strided {shape}"
        want = layer(mx.contiguous(strided))
        mx.eval(y_fast, want)
        np.testing.assert_array_equal(
            np.asarray(y_fast.view(mx.uint16)), np.asarray(want.view(mx.uint16))
        )


@pytest.mark.parametrize(
    "route_name,policy,rows,bits",
    [
        ("mma_tile", "mma", 8, 4),
        ("msg_tile", "msg", 4, 8),
    ],
)
def test_mma_and_msg_multithread_replay_and_proof_failure_retirement(
    route_name, policy, rows, bits, build_layer, monkeypatch
):
    """Review-3 F1(a)/(b): mma_tile and msg_tile graphs are per-thread, release
    on worker-thread exit, and retire cleanly on Proof.fail()."""
    from omlx.formats import tile_dense, tuning

    pol = (
        tuning.MmaTilePolicy(16, 32, 8) if policy == "mma" else tuning.MsgTilePolicy(8)
    )
    monkeypatch.setattr(
        tuning,
        "lookup",
        lambda r, *a, _rn=route_name, _p=pol, **kw: _p if r == _rn else None,
    )
    layer, packed, key = _projection(build_layer, 128, 512, bits, mx.bfloat16)
    x = mx.random.normal((rows, 512), key=key).astype(mx.bfloat16)
    first = layer(x)
    mx.eval(first)
    kept = vars(layer)[linear._KEPT]
    plan = kept.dense
    assert tile_dense.route_for(plan, rows, x.dtype).backend == policy
    main_graph = plan.binding.kept.graphs.get((x.shape, x.dtype))
    assert main_graph is not None

    worker_results = {}

    def worker():
        out = layer(x)
        mx.eval(out)
        worker_results["out"] = np.asarray(out.view(mx.uint16))
        worker_results["graph"] = plan.binding.kept.graphs.get((x.shape, x.dtype))

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert worker_results["graph"] is not None
    assert worker_results["graph"] is not main_graph
    np.testing.assert_array_equal(
        np.asarray(first.view(mx.uint16)), worker_results["out"]
    )

    # Proof.fail() retires the binding's graphs and falls back cleanly.
    plan.proof.fail()
    assert plan.binding.kept is None
    if kept.dense_fast is not None:
        assert kept.dense_fast(layer, x) is None
    fallback = layer(x)
    mx.eval(fallback)
    _c2_ok(fallback, _stock(x, packed, bits), x, packed, bits)


def test_secondary_native_fast_route_adds_bias_once_when_alternating_routes(
    build_layer, monkeypatch
):
    """A biased FormatLinear whose dense routes share dense_fast (including
    mma/msg) and whose dot route binds _fast2 (dot_fast) adds bias once and
    never falls through to Python dispatch on warm calls."""
    from omlx.custom_kernels.quant_matmul import fast as native
    from omlx.formats import tile_dense, tuning

    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires the native warm dispatcher")
    k, n = 512, 128
    mma_pol = tuning.MmaTilePolicy(16, 32, 8)
    dense_pol = tuning.DenseTilePolicy("steel", 32, 32, 32, 2, 2)

    def _lookup(route, *args, rows=1, **kw):
        if route == "mma_tile" and rows == 8:
            return mma_pol
        if route == "dense_tile" and rows >= 33:
            return dense_pol
        return None

    monkeypatch.setattr(tuning, "lookup", _lookup)
    monkeypatch.setattr(
        tuning,
        "row_bounds",
        lambda route, **kw: (8, 16, 32) if route == "mma_tile" else (16, 32),
    )

    key = mx.random.key(919)
    w = (mx.random.normal((n, k), key=key) * 0.05).astype(mx.bfloat16)
    packed = mx.quantize(w, group_size=64, bits=4)
    bias = mx.full((n,), 10.0, dtype=mx.bfloat16)
    layer = build_layer(
        "affine", 4, 64, k, dict(zip(("weight", "scales", "biases"), packed))
    )
    layer["bias"] = bias

    # Bind self._fast to dense_fast via rows=33 first.
    x33 = mx.random.normal((33, k), key=key).astype(mx.bfloat16)
    y33_cold = layer(x33)
    y33_warm = layer(x33)
    mx.eval(y33_cold, y33_warm)
    kept = vars(layer)[linear._KEPT]
    assert layer._fast is kept.dense_fast

    # Force a dot route at rows=2 into the constructed schedule as well, so
    # both mma (sharing dense_fast) and dot (secondary dot_fast -> _fast2) fire.
    plan = kept.dense
    orig = plan.schedules[mx.bfloat16]
    plan.schedules[mx.bfloat16] = (
        tile_dense.DenseRoute(2, "dot", None),
        tile_dense.DenseRoute(8, "mma", mma_pol),
        *tuple(r for r in orig if r.hi is None or r.hi > 8),
    )

    for rows, is_dot in ((8, False), (2, True)):
        x = mx.random.normal((rows, k), key=key).astype(mx.bfloat16)
        cold = layer(x)
        mx.eval(cold)
        assert layer._fast is kept.dense_fast
        if is_dot:
            assert kept.dot_fast is not None and kept.dot_fast is not layer._fast
            assert layer._fast2 is kept.dot_fast
        else:
            assert layer._fast2 is None
        with monkeypatch.context() as patch:
            patch.setattr(
                FormatLinear,
                "_scheduled",
                lambda *a, **kw: pytest.fail("warm call missed C++ fast slots"),
            )
            patch.setattr(
                FormatLinear,
                "_dot_rows",
                lambda *a, **kw: pytest.fail("warm dot call missed _fast2"),
            )
            warm = layer(x)
            mx.eval(warm)
        np.testing.assert_array_equal(
            np.asarray(cold.view(mx.uint16)),
            np.asarray(warm.view(mx.uint16)),
        )
        ref = _stock(x, packed, 4) + bias
        mx.eval(ref)
        diff = mx.max(mx.abs(warm.astype(mx.float32) - ref.astype(mx.float32))).item()
        assert diff < 1.0, f"rows={rows} double-added bias? max diff={diff}"


def test_lm_head_schedule_passes_c2_and_preserves_fast_identity(build_layer):
    """Commit 3 Test #4: (248320, 2560) 8b g64 lm_head resolves its measured
    schedule across (1, M, 2560) and (4, 5, 2560), meets strict FP32-ref C2,
    and never overwrites layer._fast when alternating row counts."""
    from omlx.formats import tile_dense, tuning

    tuning.reset_cache()
    n, k = 248320, 2560
    key = mx.random.key(402)
    # Build packed 8-bit g64 tensors directly without materializing a full
    # fp32 dense weight matrix (low peak memory on all hosts).
    weight = mx.random.randint(0, 1 << 30, shape=(n, k // 4), key=key).astype(mx.uint32)
    scales = (mx.abs(mx.random.normal((n, k // 64), key=key)) * 0.01 + 0.002).astype(
        mx.bfloat16
    )
    biases = (mx.random.normal((n, k // 64), key=key) * 0.005).astype(mx.bfloat16)
    mx.eval(weight, scales, biases)
    layer = build_layer(
        "affine", 8, 64, k, {"weight": weight, "scales": scales, "biases": biases}
    )

    # Check (1, 8, 2560) (verify msg/tile) and (4, 5, 2560) (20-row dense).
    x8 = (mx.random.normal((1, 8, k), key=key) * 0.5).astype(mx.bfloat16)
    x20 = (mx.random.normal((4, 5, k), key=key) * 0.5).astype(mx.bfloat16)
    x1 = (mx.random.normal((1, 1, k), key=key) * 0.5).astype(mx.bfloat16)
    x4 = (mx.random.normal((1, 4, k), key=key) * 0.5).astype(mx.bfloat16)
    x33 = (mx.random.normal((1, 33, k), key=key) * 0.5).astype(mx.bfloat16)
    mx.eval(x8, x20, x1, x4, x33)

    y20 = layer(x20)
    mx.eval(y20)
    kept = vars(layer)[linear._KEPT]
    plan = kept.dense
    assert plan is not None
    assert tile_dense.route_for(plan, 20, mx.bfloat16).backend in (
        "nax",
        "steel",
        "mlx",
        "mma",
    )
    first_fast = layer._fast
    assert first_fast is not None

    y8 = layer(x8)
    mx.eval(y8)
    assert tile_dense.route_for(plan, 8, mx.bfloat16).backend in (
        "msg",
        "wide",
        "mlx",
        "dot",
        "mma",
        "nax",
        "steel",
    )
    # Strict FP32-ref C2 on both (1, 8, K) and (4, 5, K) against MLX.
    for x, got in ((x8, y8), (x20, y20)):
        want = mx.quantized_matmul(
            x, weight, scales, biases, transpose=True, group_size=64, bits=8
        )
        # Check both head and tail 4096-column slices (catches huge-N indexing
        # bugs while keeping peak FP32 reference memory tiny).
        for sl in (slice(0, 4096), slice(n - 4096, n)):
            w_slice = mx.dequantize(
                weight[sl],
                scales[sl].astype(mx.float32),
                biases[sl].astype(mx.float32),
                group_size=64,
                bits=8,
            )
            ref = x.astype(mx.float32) @ w_slice.T
            mx.eval(want, ref)
            g_err = (
                mx.sqrt(mx.sum((got[..., sl].astype(mx.float32) - ref) ** 2))
                / mx.sqrt(mx.sum(ref**2))
            ).item()
            w_err = (
                mx.sqrt(mx.sum((want[..., sl].astype(mx.float32) - ref) ** 2))
                / mx.sqrt(mx.sum(ref**2))
            ).item()
            assert g_err <= 1.05 * w_err + 1e-6, (x.shape, sl, g_err, w_err)

    # Alternating (1,1,K) <-> (1,4,K) <-> (1,33,K) never overwrites layer._fast.
    for x in (x1, x4, x33, x8, x1, x20):
        mx.eval(layer(x))
        assert layer._fast is first_fast
    mx.clear_cache()


def test_measured_dot_bucket_routes_to_dot_plan_and_hits_fast2_without_python(
    build_layer, monkeypatch
):
    """A measured dense_tile 'dot' bucket routes 2..16 rows directly through
    _dot_rows (keeping 1-row decode bits), binds dot_fast to _fast2 without
    overwriting _fast, replays warm calls in C++ without entering Python
    dispatch, and clears both _fast and _fast2 on weight mutation."""
    from omlx.custom_kernels.quant_matmul import fast as native
    from omlx.formats import tile_dense, tuning

    k, n = 512, 128
    dot_pol = tuning.DenseTilePolicy("dot")
    mlx_pol = tuning.DenseTilePolicy("mlx")

    def _lookup(route, *args, rows=1, **kw):
        if route != "dense_tile":
            return None
        if rows <= 2:
            return mlx_pol
        if rows <= 4:
            return dot_pol
        return None

    monkeypatch.setattr(tuning, "lookup", _lookup)
    monkeypatch.setattr(
        tuning,
        "row_bounds",
        lambda route, **kw: (2, 4, 16) if route == "dense_tile" else (16,),
    )
    layer, packed, key = _projection(build_layer, n, k, 8, mx.bfloat16)
    bias = mx.full((n,), 2.5, dtype=mx.bfloat16)
    layer["bias"] = bias
    x2 = (mx.random.normal((2, 1, k), key=key) * 0.5).astype(mx.bfloat16)
    x4 = (mx.random.normal((4, 1, k), key=key) * 0.5).astype(mx.bfloat16)
    mx.eval(x2, x4)

    # Warm mlx at 2 streams first (binds _fast = dense_fast).
    y2_cold = layer(x2)
    mx.eval(y2_cold)
    kept = vars(layer)[linear._KEPT]
    assert tile_dense.route_for(kept.dense, 2, x2.dtype).backend == "mlx"
    assert tile_dense.route_for(kept.dense, 4, x4.dtype).backend == "dot"

    # Cold 4-stream call must dispatch directly from _scheduled -> _dot_rows
    # (never falling through tile_dense.project -> __call__ fallback).
    with monkeypatch.context() as patch:
        patch.setattr(
            tile_dense,
            "project",
            lambda *a, **kw: pytest.fail("dot route called tile_dense.project"),
        )
        y4_cold = layer(x4)
        mx.eval(y4_cold)

    # Output matches 1-row decode bit-for-bit (including single bias).
    serial = mx.concatenate([layer(x4[i : i + 1]) for i in range(4)], axis=0)
    mx.eval(serial)
    np.testing.assert_array_equal(
        np.asarray(y4_cold.view(mx.uint16)), np.asarray(serial.view(mx.uint16))
    )

    if native.has_symbol("FormatLinearFast"):
        assert layer._fast is kept.dense_fast
        assert layer._fast2 is kept.dot_fast and kept.dot_fast is not None
        with monkeypatch.context() as patch:
            patch.setattr(
                FormatLinear,
                "_scheduled",
                lambda *a, **kw: pytest.fail("warm call missed _fast/_fast2"),
            )
            patch.setattr(
                FormatLinear,
                "_dot_rows",
                lambda *a, **kw: pytest.fail("warm dot call missed _fast2"),
            )
            for _ in range(2):
                y2_warm = layer(x2)
                y4_warm = layer(x4)
                mx.eval(y2_warm, y4_warm)
                np.testing.assert_array_equal(
                    np.asarray(y2_cold.view(mx.uint16)),
                    np.asarray(y2_warm.view(mx.uint16)),
                )
                np.testing.assert_array_equal(
                    np.asarray(y4_cold.view(mx.uint16)),
                    np.asarray(y4_warm.view(mx.uint16)),
                )
        # Mutating a tensor clears both _fast and _fast2.
        layer["weight"] = layer["weight"]
        assert layer._fast is None and layer._fast2 is None


def test_mma_and_msg_share_dense_fast_after_mlx_warm_without_secondary_miss(
    build_layer, monkeypatch
):
    """After warming mlx (dense_fast) at 2 streams, msg_tile at 4 streams and
    mma_tile at 8 rows register onto the SAME dense_fast dispatcher (inlining
    N_size inside their compiled graphs) so all warm calls hit layer._fast."""
    from omlx.custom_kernels.quant_matmul import fast as native
    from omlx.formats import tile_dense, tuning

    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires the native warm dispatcher")
    k, n = 512, 128
    mlx_pol = tuning.DenseTilePolicy("mlx")
    msg_pol = tuning.MsgTilePolicy(4)
    mma_pol = tuning.MmaTilePolicy(16, 32, 8)

    def _lookup(route, *args, rows=1, **kw):
        if route == "dense_tile" and rows <= 2:
            return mlx_pol
        if route == "msg_tile" and 3 <= rows <= 4:
            return msg_pol
        if route == "mma_tile" and 5 <= rows <= 8:
            return mma_pol
        return None

    monkeypatch.setattr(tuning, "lookup", _lookup)
    monkeypatch.setattr(
        tuning,
        "row_bounds",
        lambda route, **kw: (2, 4, 8, 16),
    )
    layer, packed, key = _projection(build_layer, n, k, 8, mx.bfloat16)
    bias = mx.full((n,), 1.5, dtype=mx.bfloat16)
    layer["bias"] = bias
    x2 = (mx.random.normal((2, 1, k), key=key) * 0.5).astype(mx.bfloat16)
    x4 = (mx.random.normal((4, 1, k), key=key) * 0.5).astype(mx.bfloat16)
    x8 = (mx.random.normal((8, k), key=key) * 0.5).astype(mx.bfloat16)
    mx.eval(x2, x4, x8)

    cold = {tag: layer(x) for tag, x in (("2", x2), ("4", x4), ("8", x8))}
    mx.eval(*cold.values())
    kept = vars(layer)[linear._KEPT]
    assert layer._fast is kept.dense_fast and layer._fast2 is None
    assert not hasattr(kept, "dense_fast_nsize")
    mlx_keys = [
        k_g
        for k_g in kept.dense.proof.graphs
        if isinstance(k_g, tuple) and k_g and k_g[0] == "mlx"
    ]
    assert mlx_keys == [("mlx", 8, 64)]
    with monkeypatch.context() as patch:
        patch.setattr(
            FormatLinear,
            "_scheduled",
            lambda *a, **kw: pytest.fail("warm msg/mma missed dense_fast"),
        )
        patch.setattr(
            tile_dense,
            "project",
            lambda *a, **kw: pytest.fail("warm msg/mma projected"),
        )
        for tag, x in (("2", x2), ("4", x4), ("8", x8)):
            warm = layer(x)
            mx.eval(warm)
            np.testing.assert_array_equal(
                np.asarray(cold[tag].view(mx.uint16)),
                np.asarray(warm.view(mx.uint16)),
            )


def test_planless_layer_rejects_measured_dot_bucket(build_layer, monkeypatch):
    """A layer without a usable Dot plan (dot-less plugin or stock qmv_quad
    with no Dot plan) refuses a measured 'dot' bucket at construction instead
    of shadowing wide/mlx and falling back to full-matrix _decoded()
    (review pre-kit-dot-7dc4244f MEDIUM-2)."""
    from omlx.formats import affine, tile_dense, tuning

    class _DotlessPlugin(formats.QuantFormat):
        mode = "test_dotless_reject_dot"
        codes_name = "weight"
        tensors = ("scales", "biases")

        def layout(self, k, n, bits, group_size):
            return {
                "weight": ((n, k * bits // 32), mx.uint32),
                "scales": ((n, k // group_size), None),
                "biases": ((n, k // group_size), None),
            }

        def decode(self, tensors, bits, group_size):
            return mx.dequantize(
                tensors["weight"],
                tensors["scales"],
                tensors["biases"],
                group_size=group_size,
                bits=bits,
            )

        def dot(self, layer, k, n):
            return None

        def tile(self, layer, k, n):
            return affine.tile_face(layer.bits, layer.group_size, layer["scales"].dtype)

    dot_pol = tuning.DenseTilePolicy("dot")
    monkeypatch.setattr(
        tuning,
        "lookup",
        lambda route, *a, rows=1, **kw: (
            dot_pol if route == "dense_tile" and rows <= 16 else None
        ),
    )
    k, n = 256, 64
    key = mx.random.key(777)
    w = (mx.random.normal((n, k), key=key) * 0.05).astype(mx.bfloat16)
    packed = mx.quantize(w, group_size=64, bits=4)
    layer = build_layer(
        _DotlessPlugin(),
        4,
        64,
        k,
        dict(zip(("weight", "scales", "biases"), packed)),
    )
    monkeypatch.setattr(
        FormatLinear,
        "_decoded",
        lambda *a, **kw: pytest.fail("planless layer fell back to _decoded()"),
    )
    x = (mx.random.normal((8, k), key=key) * 0.5).astype(mx.bfloat16)
    y = layer(x)
    mx.eval(y)
    kept = vars(layer)[linear._KEPT]
    assert kept.plan is None
    assert tile_dense.route_for(kept.dense, 8, x.dtype).backend == "wide"

    # Stock qmv_quad (k=128, group_size=128, bits=8 -> _plan is None) also
    # rejects measured 'dot' and takes 'mlx' instead of bypassing dense_fast.
    w_q = (mx.random.normal((64, 128), key=key) * 0.05).astype(mx.bfloat16)
    packed_q = mx.quantize(w_q, group_size=128, bits=8)
    quad = build_layer(
        "affine",
        8,
        128,
        128,
        dict(zip(("weight", "scales", "biases"), packed_q)),
    )
    xq = (mx.random.normal((4, 128), key=key) * 0.5).astype(mx.bfloat16)
    yq = quad(xq)
    mx.eval(yq)
    kept_q = vars(quad)[linear._KEPT]
    assert kept_q.plan is None
    assert tile_dense.route_for(kept_q.dense, 4, xq.dtype).backend == "mlx"


@pytest.mark.parametrize("backend", ["mma", "msg"])
def test_mma_and_msg_compiled_graph_holds_no_plan_closure(
    backend, build_layer, monkeypatch
):
    """pre-kit-82917ced HIGH 1: mma/msg compiled graphs kept in global _PROOFS
    must bind only scalar descriptors and never close over DensePlan, route,
    or policy, so unloading a layer frees its DensePlan, binding, and weights."""
    from omlx.formats import tile_dense, tuning

    k, n = 512, 128
    if backend == "mma":
        pol = tuning.MmaTilePolicy(16, 32, 8)
        route_name = "mma_tile"
        rows = 8
    else:
        pol = tuning.MsgTilePolicy(4)
        route_name = "msg_tile"
        rows = 4

    monkeypatch.setattr(
        tuning,
        "lookup",
        lambda route, *a, rows=1, **kw: (
            pol if route == route_name and rows <= 16 else None
        ),
    )
    monkeypatch.setattr(tuning, "row_bounds", lambda route, **kw: (4, 8, 16))
    layer, _, key = _projection(build_layer, n, k, 8, mx.bfloat16)
    x = (mx.random.normal((rows, k), key=key) * 0.5).astype(mx.bfloat16)
    mx.eval(x)

    y = layer(x)
    mx.eval(y)
    kept = vars(layer)[linear._KEPT]
    assert tile_dense.route_for(kept.dense, rows, x.dtype).backend == backend
    ref_binding = weakref.ref(kept.dense.binding)
    del kept, layer, y
    gc.collect()
    assert ref_binding() is None, f"{backend} compiled graph pinned DensePlan.binding"


def test_publish_fast_ignores_stale_kept_after_mutation(build_layer, monkeypatch):
    """pre-kit-82917ced MEDIUM 2: a stale _register_fast / _register_dot_fast
    from a pre-mutation _Kept must not strand an obsolete dispatcher in _fast
    after weight mutation."""
    from omlx.custom_kernels.quant_matmul import fast as native
    from omlx.formats import tile_dense, tuning

    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires the native warm dispatcher")

    k, n = 256, 128
    mlx_pol = tuning.DenseTilePolicy("mlx")
    dot_pol = tuning.DenseTilePolicy("dot")
    monkeypatch.setattr(
        tuning,
        "lookup",
        lambda route, *a, rows=1, **kw: (
            mlx_pol
            if route == "dense_tile" and rows == 2
            else dot_pol if route == "dense_tile" and rows == 4 else None
        ),
    )
    monkeypatch.setattr(tuning, "row_bounds", lambda route, **kw: (2, 4, 16))
    layer, _, key = _projection(build_layer, n, k, 8, mx.bfloat16)
    x2 = (mx.random.normal((2, 1, k), key=key) * 0.5).astype(mx.bfloat16)
    x4 = (mx.random.normal((4, 1, k), key=key) * 0.5).astype(mx.bfloat16)
    mx.eval(x2, x4)

    mx.eval(layer(x2), layer(x4))
    old_kept = vars(layer)[linear._KEPT]
    old_fast = old_kept.dense_fast
    old_dot = old_kept.dot_fast
    assert old_fast is not None and old_dot is not None
    assert layer._fast is old_fast and layer._fast2 is old_dot

    # Mutate weight: clears _KEPT, _FAST, _FAST2; then replay stale registrations.
    layer["weight"] = mx.array(layer["weight"])
    layer._register_fast(old_kept, old_kept.dense, (x2.shape, x2.dtype))
    layer._register_dot_fast(old_kept, old_kept.plan, (x4.shape, x4.dtype))
    assert layer._fast is None and layer._fast2 is None

    mx.eval(layer(x2), layer(x4))
    new_kept = vars(layer)[linear._KEPT]
    assert new_kept is not old_kept
    assert layer._fast is new_kept.dense_fast
    assert layer._fast is not old_fast
    assert layer._fast2 is new_kept.dot_fast
    assert layer._fast2 is not old_dot
    assert tile_dense.route_for(new_kept.dense, 2, x2.dtype).backend == "mlx"


def test_all_mutators_clear_fast2_slot(build_layer):
    """Kill pre-kit-82917ced surviving mutants: every layer mutator clears _fast2."""
    k, n = 256, 64
    layer, _, _ = _projection(build_layer, n, k, 8, mx.bfloat16)
    sentinel = object()

    def _plant():
        layer.__dict__[linear._FAST] = sentinel
        layer.__dict__[linear._FAST2] = sentinel

    # __delitem__
    layer["extra"] = mx.zeros((1,), dtype=mx.bfloat16)
    _plant()
    del layer["extra"]
    assert layer._fast is None and layer._fast2 is None

    # pop
    layer["extra"] = mx.zeros((1,), dtype=mx.bfloat16)
    _plant()
    layer.pop("extra")
    assert layer._fast is None and layer._fast2 is None

    # setdefault
    _plant()
    layer.setdefault("extra", mx.zeros((1,), dtype=mx.bfloat16))
    assert layer._fast is None and layer._fast2 is None

    # popitem
    _plant()
    k_pop, v_pop = layer.popitem()
    assert layer._fast is None and layer._fast2 is None
    layer[k_pop] = v_pop

    # __delattr__
    layer.temp_attr = 1
    _plant()
    del layer.temp_attr
    assert layer._fast is None and layer._fast2 is None

    # _keep replacement when weight changed without going through __setitem__
    kept = layer._keep()
    _plant()
    dict.__setitem__(layer, "weight", mx.array(layer["weight"]))
    kept2 = layer._keep()
    assert kept2 is not kept
    assert layer._fast is None and layer._fast2 is None

    # clear
    _plant()
    layer.clear()
    assert layer._fast is None and layer._fast2 is None


def test_dot_schedule_guards_vpt_exact_dtype_and_max_rows(build_layer, monkeypatch):
    """Kill pre-kit-82917ced surviving mutants on tile_dense.py:1145, 1284, 1296."""
    from dataclasses import replace

    from omlx.formats import tile_dense, tuning

    dot_pol = tuning.DenseTilePolicy("dot")
    monkeypatch.setattr(
        tuning,
        "lookup",
        lambda route, *a, rows=1, **kw: dot_pol if route == "dense_tile" else None,
    )
    monkeypatch.setattr(tuning, "row_bounds", lambda route, **kw: (4, 16, 24))
    key = mx.random.key(909)

    # 1. tile_dense.py:1145: bound > _DOT_MAX_ROWS rejects measured 'dot' past 16 rows.
    layer, _, _ = _projection(build_layer, 64, 256, 8, mx.bfloat16)
    plan = tile_dense.build_plan(layer, tensor_units=False)
    assert plan is not None
    assert tile_dense.route_for(plan, 16, mx.bfloat16).backend == "dot"
    assert tile_dense.route_for(plan, 24, mx.bfloat16).backend != "dot"

    # 2. tile_dense.py:1284: n % 8 != 0 with dot_face.vpt < 1 rejects measured 'dot'.
    w_odd = (mx.random.normal((65, 256), key=key) * 0.05).astype(mx.bfloat16)
    packed_odd = mx.quantize(w_odd, group_size=64, bits=4)
    odd_layer = build_layer(
        "affine",
        4,
        64,
        256,
        dict(zip(("weight", "scales", "biases"), packed_odd)),
    )
    real_dot = formats.dot(odd_layer, 256, 65)
    assert real_dot is not None
    vpt0_dot = replace(real_dot, vpt=0)
    with monkeypatch.context() as patch:
        patch.setattr(formats, "dot", lambda *a, **kw: vpt0_dot)
        plan_vpt0 = tile_dense.build_plan(odd_layer, tensor_units=False)
        assert plan_vpt0 is not None
        assert tile_dense.route_for(plan_vpt0, 4, mx.bfloat16).backend != "dot"

    # 3. tile_dense.py:1296: dot_face.exact_for(dt) gates measured 'dot' per dtype
    # in both uncached and _schedule_cache paths.
    real_tile = formats.tile(layer, 256, 64)
    assert real_tile is not None
    dual_tile = replace(real_tile, dtypes=(mx.bfloat16, mx.float16))
    bf16_only_dot = replace(real_dot, dtypes=(mx.bfloat16,))
    with monkeypatch.context() as patch:
        patch.setattr(formats, "tile", lambda *a, **kw: dual_tile)
        for scache in (None, {}):
            plan_dt = tile_dense.build_plan(
                layer,
                tensor_units=False,
                dot_face=bf16_only_dot,
                _schedule_cache=scache,
            )
            assert plan_dt is not None
            assert tile_dense.route_for(plan_dt, 4, mx.bfloat16).backend == "dot"
            assert tile_dense.route_for(plan_dt, 4, mx.float16).backend != "dot"
