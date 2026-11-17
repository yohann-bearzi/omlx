"""The reference a quantization in a quant format is verified against
(``omlx.oq_verify``): the source's own forward, one decoder layer at a time."""

from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import llama
from mlx_lm.utils import load_model

from omlx import oq_verify

_CONFIG = {
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 3,
    "intermediate_size": 128,
    "num_attention_heads": 2,
    "rms_norm_eps": 1e-5,
    "vocab_size": 64,
}


@pytest.fixture
def checkpoint(tmp_path):
    """A tiny Llama checkpoint of random weights."""
    mx.random.seed(0)
    model = llama.Model(llama.ModelArgs.from_dict(_CONFIG))
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    (tmp_path / "config.json").write_text(json.dumps(_CONFIG))
    return tmp_path


def _prompt():
    return mx.arange(20, dtype=mx.int32) % _CONFIG["vocab_size"]


def test_the_streamed_reference_is_the_resident_model_s(checkpoint):
    lazy, _ = load_model(checkpoint, lazy=True)
    reference = oq_verify.reference_of(lazy, _prompt(), lazy.model.layers)
    resident, _ = load_model(checkpoint)
    logits = resident(_prompt()[None])[0].astype(mx.float32)
    logprobs = np.array(logits - mx.logsumexp(logits, axis=-1, keepdims=True))
    order = np.argsort(-logprobs, axis=-1, kind="stable")
    top = order[:, : oq_verify.REFERENCE_TOP]
    assert reference["prompt"].tolist() == _prompt().tolist()
    assert reference["ids"].dtype == np.int32 and reference["ids"].shape == (20, 16)
    np.testing.assert_array_equal(
        np.take_along_axis(logprobs, reference["ids"], axis=-1),
        np.take_along_axis(logprobs, top, axis=-1),
    )
    np.testing.assert_allclose(
        reference["logprobs"], np.take_along_axis(logprobs, top, axis=-1), atol=1e-6
    )
    rest = np.take_along_axis(logprobs, order[:, oq_verify.REFERENCE_TOP :], axis=-1)
    np.testing.assert_allclose(
        reference["tail"], np.log(np.exp(rest).sum(axis=-1)), atol=1e-5
    )


def test_a_layer_s_weights_go_once_it_has_run(checkpoint):
    model, _ = load_model(checkpoint, lazy=True)
    layers = model.model.layers
    held = []
    attention = type(layers[0].self_attn)
    call = attention.__call__

    def counting(self, *args, **kwargs):
        # When a layer runs, the layers before it hold no weights.
        held.append(
            [
                sum(v.size for _, v in tree_flatten(layer.parameters()))
                for layer in layers
            ]
        )
        return call(self, *args, **kwargs)

    attention.__call__ = counting
    try:
        oq_verify.reference_of(model, _prompt(), layers)
    finally:
        attention.__call__ = call
    assert [[size == 0 for size in sizes] for sizes in held] == [
        [False, False, False],
        [True, False, False],
        [True, True, False],
    ]
    assert all(
        v.size == 0 for layer in layers for _, v in tree_flatten(layer.parameters())
    )


def test_the_pass_holds_one_layer_s_weights_at_a_time(tmp_path):
    config = {**_CONFIG, "hidden_size": 256, "intermediate_size": 2048}
    config["num_hidden_layers"] = 6
    mx.random.seed(0)
    model = llama.Model(llama.ModelArgs.from_dict(config))
    weights = dict(tree_flatten(model.parameters()))
    layer = sum(v.nbytes for k, v in weights.items() if k.startswith("model.layers.0."))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(config))
    del model, weights
    mx.clear_cache()
    lazy, _ = load_model(tmp_path, lazy=True)
    mx.reset_peak_memory()
    before = mx.get_active_memory()
    oq_verify.reference_of(lazy, _prompt(), lazy.model.layers)
    # The six layers would be four times this.
    assert mx.get_peak_memory() - before < 1.5 * layer


def test_a_layer_that_runs_twice_is_refused_and_the_classes_restored(checkpoint):
    model, _ = load_model(checkpoint, lazy=True)
    layer = model.model.layers[0]
    cls = type(layer)
    own = cls.__dict__["__call__"]
    with oq_verify.layer_by_layer([layer]):
        x = mx.zeros((1, 2, _CONFIG["hidden_size"]))
        layer(x)
        with pytest.raises(RuntimeError, match="ran twice"):
            layer(x)
        # Another layer of the class runs as it does.
        other = model.model.layers[1]
        before = sum(v.size for _, v in tree_flatten(other.parameters()))
        other(x)
        assert sum(v.size for _, v in tree_flatten(other.parameters())) == before
    assert cls.__dict__["__call__"] is own


def test_a_class_without_its_own_call_gets_none_back():
    class Base(nn.Module):
        def __call__(self, x):
            return x + 1

    class Layer(Base):
        pass

    layer = Layer()
    with oq_verify.layer_by_layer([layer]):
        assert "__call__" in Layer.__dict__
        layer(mx.array(1))
    assert "__call__" not in Layer.__dict__
