# SPDX-License-Identifier: Apache-2.0
"""Boundaries of the native warm dispatcher and its Qwen family selector."""

import mlx.core as mx
import pytest

from omlx.custom_kernels.quant_matmul import fast as native
from omlx.formats import kernels, linear
from omlx.patches.format_load import loading


def dispatcher(weight, operands, kept):
    if not native.has_symbol("FormatLinearFast"):
        pytest.skip("requires the native warm dispatcher")
    return native.format_linear_fast(weight, None, operands, {"_kept": kept}, "weight")


@pytest.mark.parametrize("count", (15, 16, 17))
def test_native_operand_limit_includes_the_input(count):
    weight, layer_input = object(), mx.array([1.0])
    layer = {"weight": weight}
    operands = [weight] * count
    key = layer_input.shape, layer_input.dtype
    proof = kernels.Proof()
    if count >= 16:
        # Observe registration before the native code could fill an oversized
        # argument vector, even when run against the off-by-one mutant.
        def unexpected_lookup(signature):
            pytest.fail(f"registered {count} static operands for {signature}")
        proof._kept.replay = unexpected_lookup
    else:
        proof.keep(key, lambda *args: args)
    replay = dispatcher(weight, operands, proof._kept)
    replay.__self__.register(key)
    result = replay(layer, layer_input)
    if count < 16:
        assert len(result) == 16 and result[0] is layer_input
        assert all(operand is weight for operand in result[1:])
    else:
        assert result is None


def test_registering_the_65th_shape_restarts_native_signature_lookup():
    # Keep the worker lookup observable independently of Proof.keep's own
    # 64-graph eviction. An unregistered shape must never reach that lookup.
    class Lookups:
        def __init__(self):
            self.keys = []

        def replay(self, key):
            self.keys.append(key)
            return None

    kept, weight = Lookups(), object()
    replay = dispatcher(weight, [weight], kept)
    inputs = [mx.zeros((m, 1)) for m in range(1, 66)]
    keys = [(x.shape, x.dtype) for x in inputs]
    for key in keys[:64]:
        replay.__self__.register(key)
    layer = {"weight": weight}
    for i in (0, 63):
        assert replay(layer, inputs[i]) is None
    assert kept.keys == [keys[0], keys[63]]
    replay.__self__.register(keys[-1])
    kept.keys.clear()
    for i in (0, 63, 64):
        assert replay(layer, inputs[i]) is None
    assert kept.keys == [keys[-1]]


def test_damaged_checkpoint_config_does_not_select_qwen(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "qwen4_exp"')
    with loading(tmp_path):
        assert not linear._loading_qwen()
