"""The reference a quantization in a quant format is verified against
(``omlx.oq_verify``): the source's own forward, one decoder layer at a time."""

from __future__ import annotations

import json
import sys

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


def test_streamed_subclass_releases_weights_after_the_outermost_call():
    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.array([2.0])

        def __call__(self, x):
            return x * self.weight

    class Sub(Base):
        def __call__(self, x):
            return super().__call__(x) + self.weight

    base, sub = Base(), Sub()
    with oq_verify.layer_by_layer([base, sub]):
        assert sub(mx.array([3.0])).tolist() == [8.0]
        assert sub.weight.size == 0
        assert base(mx.array([4.0])).tolist() == [8.0]
        assert base.weight.size == 0


def test_overlapping_reference_passes_restore_the_shared_layer_class():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    class Layer(nn.Module):
        def __call__(self, x):
            return x

    original = Layer.__call__
    started, entered, finish = Event(), Event(), Event()

    def other_pass():
        started.set()
        with oq_verify.layer_by_layer([Layer()]):
            entered.set()
            assert finish.wait(10)

    with ThreadPoolExecutor(1) as workers:
        try:
            with oq_verify.layer_by_layer([Layer()]):
                future = workers.submit(other_pass)
                assert started.wait(10)
                # A serialized pass can enter as soon as this first pass exits.
                entered.wait(0.5)
        finally:
            finish.set()
        future.result(timeout=10)
    assert Layer.__call__ is original


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


@pytest.mark.parametrize("stacked", [False, True], ids=["linear", "switch"])
def test_streamed_decoder_releases_a_nested_format_projection(stacked, monkeypatch):
    from omlx.formats import testing
    from omlx.patches import module_cache
    from tests.test_formats_linear import _experts, _pair, _switch_pair, _x

    monkeypatch.setattr(module_cache, "_SLOTS", set(module_cache._SLOTS))

    class Block(nn.Module):
        def __call__(self, *args):
            module_cache.cached_per_module(self, "_test_block", lambda m: m.proj.weight)
            return self.proj(*args)

    class Decoder(nn.Module):
        def __call__(self, *args):
            module_cache.cached_per_module(
                self, "_test_decoder", lambda m: m.mlp.proj.weight, depth=2
            )
            return self.mlp(*args)

    decoder = Decoder()
    decoder.mlp = Block()
    with testing.verified("llama"):
        resident, decoder.mlp.proj = (_switch_pair if stacked else _pair)(
            320, 72, bias=True
        )
    args = (_x((2, 1, 320)), _experts((2,))) if stacked else (_x((2, 320)),)
    expected = resident(*args)
    mx.eval(expected)
    del resident
    names = ("weight", "scales", "biases", "bias")
    orig = {name: decoder.mlp.proj[name] for name in names}
    assert all(orig[name].size > 0 for name in names)
    with oq_verify.layer_by_layer([decoder]):
        actual = decoder(*args)
        assert mx.array_equal(actual, expected).item()
        assert {name: decoder.mlp.proj[name].size for name in names} == dict.fromkeys(
            names, 0
        )
        assert {name: sys.getrefcount(orig[name]) for name in names} == dict.fromkeys(
            names, 2
        )


def test_streamed_decoder_releases_custom_format_declared_tensors(monkeypatch):
    from omlx import formats
    from omlx.formats import testing
    from omlx.formats.linear import FormatLinear

    class _StreamToy(formats.QuantFormat):
        mode, tensors, contract = "toy_stream", ("lut",), formats.CONTRACT

        def decode(self, tensors, bits, group_size):
            return tensors["weight"].astype(mx.float32) * mx.expand_dims(
                tensors["lut"], -1
            )

    formats.load()
    monkeypatch.setitem(formats._FORMATS, "toy_stream", _StreamToy())

    class Block(nn.Module):
        def __call__(self, x):
            return self.proj(x)

    class Decoder(nn.Module):
        def __call__(self, x):
            return self.mlp(x)

    mx.random.seed(0)
    weight = mx.random.randint(0, 16, (16, 32), dtype=mx.uint8)
    lut = mx.random.normal((16,)).astype(mx.float32)
    mx.eval(weight, lut)
    with testing.verified("llama"):
        proj = FormatLinear("toy_stream", 4, 32, 32, {"weight": weight, "lut": lut})
        partial = FormatLinear(
            "toy_stream",
            4,
            32,
            32,
            {"weight": mx.zeros((16, 32), dtype=mx.uint8), "lut": mx.zeros((16,))},
        )
    del partial["lut"]
    aux = nn.Module()
    proj.aux = aux
    decoder = Decoder()
    decoder.mlp = Block()
    decoder.mlp.proj = proj
    decoder.mlp.partial = partial
    orig_w, orig_lut = proj["weight"], proj["lut"]
    del weight, lut
    with oq_verify.layer_by_layer([decoder]):
        out = decoder(mx.ones((2, 32), dtype=mx.float32))
        mx.eval(out)
        assert proj["weight"].size == 0
        assert proj["lut"].size == 0
        assert "lut" not in partial
        assert partial["weight"].size == 0
        assert proj.aux is aux
        assert sys.getrefcount(orig_w) == 2
        assert sys.getrefcount(orig_lut) == 2


def test_a_published_cache_entry_can_already_be_cleared(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, current_thread

    from omlx.patches import module_cache

    mod = nn.Linear(16, 8, bias=False)
    slot = "_test_first_cache"
    caller = current_thread()
    ready, resume = Event(), Event()
    lock = module_cache._SLOTS_LOCK

    class PauseRegistration:
        def __enter__(self):
            if current_thread() is not caller:
                ready.set()
                assert resume.wait(10)
            lock.acquire()

        def __exit__(self, *exc):
            lock.release()

    class Slots(set):
        def __iter__(self):
            # A clear takes a stable snapshot while registrations are excluded.
            assert lock.locked()
            return super().__iter__()

    monkeypatch.setattr(module_cache, "_SLOTS", Slots())
    monkeypatch.setattr(module_cache, "_SLOTS_LOCK", PauseRegistration())
    with ThreadPoolExecutor(1) as pool:
        build = pool.submit(
            module_cache.cached_per_module, mod, slot, lambda m: m.weight
        )
        try:
            assert ready.wait(10)
            published = slot in mod.__dict__
            if published:
                module_cache.clear_cached(mod)
        finally:
            resume.set()
        assert build.result(timeout=10) is mod.weight
        if not published:
            assert slot in mod.__dict__
            module_cache.clear_cached(mod)
        assert slot not in mod.__dict__

        # A miss for a known slot can rebuild without taking the registry lock.
        ready.clear()
        rebuild = pool.submit(
            module_cache.cached_per_module, mod, slot, lambda m: m.weight
        )
        assert rebuild.result(timeout=10) is mod.weight
        assert not ready.is_set()
        assert slot in mod.__dict__


def test_failed_cache_build_does_not_register_or_publish_a_slot(monkeypatch):
    from omlx.patches import module_cache

    monkeypatch.setattr(module_cache, "_SLOTS", set())
    mod = nn.Linear(16, 8, bias=False)
    slot = "_test_failed_build"

    def fail(module):
        raise RuntimeError("failed to build the cache value")

    with pytest.raises(RuntimeError, match="failed to build the cache value"):
        module_cache.cached_per_module(mod, slot, fail)
    assert slot not in module_cache._SLOTS
    assert slot not in mod.__dict__


def test_clear_cached_is_thread_safe(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from omlx.patches import module_cache
    from omlx.patches.module_cache import cached_per_module, clear_cached

    monkeypatch.setattr(module_cache, "_SLOTS", set(module_cache._SLOTS))

    def worker(idx: int) -> int:
        mod = nn.Linear(16, 8, bias=False)
        orig = mod.weight
        cached_per_module(mod, f"_omlx_test_clear_{idx}", lambda m: m.weight, depth=0)
        clear_cached(mod)
        mod.weight = mx.zeros((0,))
        return sys.getrefcount(orig)

    with ThreadPoolExecutor(4) as pool:
        refs = list(pool.map(worker, range(16)))
    assert refs == [2] * 16


def test_streamed_forward_accepts_a_one_shot_layer_iterator():
    layer = nn.Linear(32, 16)
    with oq_verify.layer_by_layer(iter([layer])):
        layer(mx.ones((1, 32)))
        assert layer.weight.size == 0


def test_a_reference_cache_replaced_by_a_file_is_refused(tmp_path):
    cache = tmp_path / "cache"
    cache.write_bytes(b"replaced directory")
    with pytest.raises(NotADirectoryError):
        oq_verify._take_reference(cache / "reference.npz", {}, None)
