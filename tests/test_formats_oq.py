# SPDX-License-Identifier: Apache-2.0
"""oQ writes a checkpoint in a quant format (``quant_format``): what a format
is offered and what is checked of its answer, then a tiny checkpoint written
in a toy format and loaded back.
"""

from __future__ import annotations

import inspect
import json
import logging
import textwrap
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_5_moe, qwen3_next
from mlx_lm.models import switch_layers as lm_switch
from mlx_lm.utils import load_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from omlx import formats, oq
from omlx.formats.linear import FormatLinear, FormatSwitchLinear
from omlx.utils.model_loading import maybe_apply_pre_load_patches
from tests.test_formats_kernels import _BLOCKS, _PAGE
from tests.test_formats_linear import _ToyFormat

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

_RECORD = "omlx_quant_formats"  # in a checkpoint's config: what reads its layers
# docs/quant-formats.md quotes what oQ says of a format, one named myformat.
_QUOTED = " ".join(_PAGE.split())


def _quoted(message, name="toy"):
    return message.replace(name, "myformat") in _QUOTED


class _Codec(_ToyFormat):
    """The toy format with its codec: a byte for each weight and a scale for
    each row, which keep a weight to half a scale. ``answer`` stands in for
    a codec that declines, or has a defect."""

    def __init__(self):
        super().__init__()
        self.offered, self.encoded, self.lays_out = [], [], True
        self.answer = lambda tensors, w: tensors
        self.accepted = []

    def layout(self, k, n, bits, group_size):
        self.offered.append((k, n, bits, group_size))
        if not self.lays_out:
            return None
        return {"weight": ((n, k), mx.uint8), "scales": ((n,), mx.float32)}

    def encode(self, w, importance, bits, group_size):
        self.encoded.append((w, importance, bits, group_size))
        rows = w.astype(mx.float32)
        scales = mx.maximum(mx.abs(rows).max(axis=-1), 1e-6) / 127.0
        codes = mx.round(rows / mx.expand_dims(scales, -1)) + 128.0
        return self.answer({"weight": codes.astype(mx.uint8), "scales": scales}, w)

    def accept(self, record):
        self.accepted.append(record)


@pytest.fixture
def toy(monkeypatch):
    """The toy codec, registered by hand."""
    formats.load()
    fmt = _Codec()
    monkeypatch.setitem(formats._FORMATS, "toy", fmt)
    return fmt


def _weights(*shape):
    mx.random.seed(0)
    return mx.random.normal(shape)


def _same(a, b):
    return a.keys() == b.keys() and all(
        (a[name].shape, a[name].dtype) == (b[name].shape, b[name].dtype)
        and mx.array_equal(a[name], b[name])
        for name in a
    )


def test_a_matrix_is_offered_whole(toy):
    w, importance = _weights(8, 64), mx.arange(64) + 1.0
    kept = oq._encode_in_format(toy, "a.weight", w, importance, 4, 32)
    assert toy.offered == [(64, 8, 4, 32)]
    [(given, weighed, bits, group_size)] = toy.encoded
    assert given.shape == (8, 64) and mx.array_equal(given, w)
    assert weighed.shape == (64,) and mx.array_equal(weighed, importance)
    assert (bits, group_size) == (4, 32)
    assert _same(kept, _Codec().encode(w, importance, 4, 32))
    # So is a stack that fits a launch, with the importance of each matrix.
    w, importance = _weights(3, 8, 64), mx.arange(3 * 64).reshape(3, 64) + 1.0
    kept = oq._encode_in_format(toy, "experts.weight", w, importance, 4, 32)
    assert toy.offered[1:] == [(64, 8, 4, 32)] and len(toy.encoded) == 2
    given, weighed = toy.encoded[1][:2]
    assert given.shape == (3, 8, 64) and weighed.shape == (3, 64)
    assert _same(kept, _Codec().encode(w, importance, 4, 32))


@pytest.mark.parametrize("own", [True, False], ids=["one each", "one for all"])
def test_a_stack_is_offered_in_runs_of_its_matrices(toy, monkeypatch, own):
    e, n, k = 5, 8, 64
    w = _weights(e, n, k)
    importance = mx.random.uniform(0.5, 2.0, (e, k) if own else (k,))
    # Two of its matrices fit a launch.
    monkeypatch.setattr(oq, "_QUANTIZE_CHUNK_BYTES", 2 * 2 * n * k)
    kept = oq._encode_in_format(toy, "experts.weight", w, importance, 3, 64)
    assert toy.offered == [(k, n, 3, 64)]
    runs = [(0, 2), (2, 4), (4, 5)]
    assert len(toy.encoded) == len(runs)
    for (r0, r1), offer in zip(runs, toy.encoded, strict=True):
        given, weighed, bits, group_size = offer
        assert given.shape == (r1 - r0, n, k) and mx.array_equal(given, w[r0:r1])
        # A run's matrices with their importances, or the one they share.
        want = importance[r0:r1] if own else importance
        assert weighed.shape == want.shape and mx.array_equal(weighed, want)
        assert (bits, group_size) == (3, 64)
    # The layer keeps the runs' tensors, joined along the stack.
    assert kept["weight"].shape == (e, n, k) and kept["scales"].shape == (e, n)
    assert _same(kept, _Codec().encode(w, importance, 3, 64))


def test_what_is_no_matrix_or_fits_no_launch_is_not_offered(toy, monkeypatch):
    importance = mx.ones((64,))

    def offer(*shape):
        return oq._encode_in_format(
            toy, "a.weight", _weights(*shape), importance, 4, 64
        )

    assert offer(2, 2, 8, 64) is None
    monkeypatch.setattr(oq, "_QUANTIZE_CHUNK_BYTES", 2 * 8 * 64 - 2)
    assert offer(8, 64) is None and offer(3, 8, 64) is None
    assert toy.encoded == []
    # A matrix of a launch's size is, and a stack of them one at a time.
    monkeypatch.setattr(oq, "_QUANTIZE_CHUNK_BYTES", 2 * 8 * 64)
    assert offer(8, 64).keys() == {"weight", "scales"}
    assert offer(3, 8, 64)["weight"].shape == (3, 8, 64)
    assert [given.shape[0] for given, *_ in toy.encoded] == [8, 1, 1, 1]
    # MLX's own limit on a launch holds whatever the buffer's.
    monkeypatch.setattr(oq, "_QUANTIZE_CHUNK_BYTES", 1 << 40)
    huge = SimpleNamespace(shape=((1 << 15) + 1, 1 << 15))
    assert oq._encode_in_format(toy, "a.weight", huge, importance, 4, 64) is None


def test_a_matrix_the_format_lays_out_none_of_is_not_offered(toy):
    toy.lays_out = False
    w = _weights(8, 64)
    assert oq._encode_in_format(toy, "a.weight", w, mx.ones((64,)), 4, 64) is None
    assert toy.offered == [(64, 8, 4, 64)] and toy.encoded == []


def test_a_format_declines_a_matrix_or_a_whole_stack(toy, monkeypatch):
    importance = mx.ones((64,))
    monkeypatch.setattr(oq, "_QUANTIZE_CHUNK_BYTES", 2 * 2 * 8 * 64)
    toy.answer = lambda tensors, w: None
    for shape in ((8, 64), (5, 8, 64)):
        w = _weights(*shape)
        assert oq._encode_in_format(toy, "a.weight", w, importance, 4, 64) is None
    # The stack's first run was the answer for the stack.
    assert [given.shape for given, *_ in toy.encoded] == [(8, 64), (2, 8, 64)]
    # A stack is one layer: a format does not keep some of its matrices.
    answers = iter([True, False])
    toy.answer = lambda tensors, w: tensors if next(answers) else None
    with pytest.raises(
        ValueError,
        match="quant format 'toy' encoded the first 2 matrices of experts.weight "
        "and none of the next: a stack is one layer",
    ):
        oq._encode_in_format(toy, "experts.weight", w, importance, 4, 64)


@pytest.mark.parametrize(
    "answer",
    [
        lambda tensors: list(tensors.values()),
        lambda tensors: {**tensors, "scales": 1.0},
    ],
    ids=["a list", "a float"],
)
def test_an_answer_that_is_no_dict_of_arrays_is_refused(toy, answer):
    toy.answer = lambda tensors, w: answer(tensors)
    with pytest.raises(
        TypeError,
        match="quant format 'toy': encode gives the tensors of a.weight as a dict "
        "of arrays by name, or None",
    ):
        oq._encode_in_format(toy, "a.weight", _weights(8, 64), mx.ones((64,)), 4, 64)


@pytest.mark.parametrize("shape", [(8, 64), (3, 8, 64)], ids=["matrix", "stack"])
@pytest.mark.parametrize(
    "answer",
    [
        lambda tensors: {**tensors, "biases": tensors["scales"]},
        lambda tensors: {"weight": tensors["weight"]},
        lambda tensors: {**tensors, "scales": tensors["scales"][..., None]},
        lambda tensors: {**tensors, "scales": tensors["scales"].astype(mx.float16)},
        lambda tensors: {name: tensor[0] for name, tensor in tensors.items()},
    ],
    ids=["a tensor more", "a tensor fewer", "a shape", "a dtype", "an axis fewer"],
)
def test_tensors_that_are_not_the_layout_s_are_refused(toy, answer, shape):
    # A checkpoint loads by the layout: these would be written, and not load.
    toy.answer = lambda tensors, w: answer(tensors)
    with pytest.raises(
        ValueError,
        match=r"quant format 'toy' encoded a\.weight as \{.*\}, and lays it out "
        r"as \{'weight': \(\(.*8, 64\), mlx\.core\.uint8\), 'scales': .*\}: the "
        "checkpoint would not load",
    ):
        oq._encode_in_format(toy, "a.weight", _weights(*shape), mx.ones((64,)), 4, 64)


def test_the_contract_page_quotes_the_refusal_of_other_tensors(toy, monkeypatch):
    monkeypatch.setattr(toy, "mode", "myformat")
    toy.answer = lambda tensors, w: {"weight": tensors["weight"]}
    name = "model.layers.0.mlp.gate_proj.weight"
    with pytest.raises(ValueError) as refusal:
        oq._encode_in_format(toy, name, _weights(16, 64), mx.ones((64,)), 8, 1)
    assert str(refusal.value) in _QUOTED


def test_the_contract_page_s_codec_is_the_toy_s():
    (text,) = (
        text
        for language, text in _BLOCKS
        if language == "python" and "def encode" in text and "rows" in text
    )
    scope = {"mx": mx}
    exec(textwrap.dedent(text), scope)
    w, importance = _weights(8, 64), mx.ones((64,))
    kept = scope["encode"](None, w, importance, 8, 1)
    assert _same(kept, _Codec().encode(w, importance, 8, 1))
    assert scope["encode"](None, w, importance, 4, 1) is None


def test_a_run_s_tensors_have_the_stack_s_axis(toy):
    toy.answer = lambda tensors, w: {name: t[0] for name, t in tensors.items()}
    with pytest.raises(ValueError, match=r"as \{'weight': \(\(1, 8, 64\), mlx"):
        oq._encode_in_format(toy, "a.weight", _weights(1, 8, 64), mx.ones((64,)), 4, 64)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """A tiny Qwen3.5 MoE checkpoint in bfloat16, with its tokenizer."""
    config = {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "vocab_size": 64,
        "full_attention_interval": 2,
        "max_position_embeddings": 512,
        "mtp_num_hidden_layers": 0,
        "tie_word_embeddings": False,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "shared_expert_intermediate_size": 128,
        "moe_intermediate_size": 128,
    }
    mx.random.seed(7)
    model = qwen3_5_moe.Model(qwen3_5_moe.ModelArgs.from_dict(config))
    model.set_dtype(mx.bfloat16)
    path = tmp_path_factory.mktemp("source")
    weights = {
        name.removeprefix("language_model."): value
        for name, value in tree_flatten(model.parameters())
    }
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    (path / "config.json").write_text(json.dumps(config))
    vocab = {f"t{i}": i for i in range(config["vocab_size"])}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="t0"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0").save_pretrained(
        path
    )
    return path


def _quantize(source, output, **kwargs):
    """oQ4e of ``source``: the output's tensors, its config, and the layers
    that oQe had an importance for."""
    tokens = (mx.arange(128).reshape(8, 16) % 63 + 1).astype(mx.int32)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(oq, "_load_calibration_data", lambda *args, **kwargs: tokens)
        oq.quantize_oq_streaming(
            str(source),
            str(output),
            4,
            enhanced=True,
            imatrix_num_samples=4,
            imatrix_seq_length=16,
            sensitivity_map_override={0: 0.1, 1: 0.2},
            **kwargs,
        )
    tensors = {}
    for shard in sorted(output.glob("*.safetensors")):
        tensors.update(mx.load(str(shard)))
    config = json.loads((output / "config.json").read_text())
    report = json.loads((output / "oq_imatrix_report.json").read_text())
    return tensors, config, sorted(report["applied"])


@pytest.fixture(scope="module")
def plain(source, tmp_path_factory):
    """What oQ4e writes of the checkpoint without a format."""
    return _quantize(source, tmp_path_factory.mktemp("plain") / "oqe")


def _in_format(config, mode="toy"):
    return {
        name: entry
        for name, entry in config["quantization"].items()
        if isinstance(entry, dict) and entry["mode"] == mode
    }


def _beside(tensors, config, layers):
    """An output without ``layers``: its other tensors, and its config
    without their entries and the formats' record."""
    tensors = {
        name: tensor
        for name, tensor in tensors.items()
        if name.rpartition(".")[0] not in layers
    }
    config = {name: value for name, value in config.items() if name != _RECORD}
    for key in ("quantization", "quantization_config"):
        config[key] = {
            name: entry for name, entry in config[key].items() if name not in layers
        }
    return tensors, config


def _assert_same_output(ours, theirs):
    (tensors, config), (their_tensors, their_config) = ours, theirs
    assert sorted(tensors) == sorted(their_tensors)
    assert _same(tensors, their_tensors)
    assert config == their_config


def test_oq_writes_the_matrices_oqe_weighs_in_the_format(
    source, plain, toy, tmp_path, monkeypatch, caplog
):
    monkeypatch.setitem(formats._ORIGINS, "toy", "omlx-toy 0.3.1")
    caplog.set_level(logging.INFO, logger="omlx.oq")
    output = tmp_path / "oqe-toy"
    tensors, config, applied = _quantize(source, output, quant_format="toy")
    entries = _in_format(config)
    # Each matrix that oQe has an importance for, the stacks of experts among
    # them, at the bits and group size that oQ gives it without the format.
    assert sorted(entries) == applied
    stacks = {name for name in entries if tensors[f"{name}.weight"].ndim == 3}
    assert len(stacks) == 6 and len(entries) > 12
    plain_tensors, plain_config, plain_applied = plain
    assert plain_applied == applied
    default = {key: plain_config["quantization"][key] for key in ("bits", "group_size")}
    weights = mx.load(str(source / "model.safetensors"))
    for name, entry in entries.items():
        theirs = plain_config["quantization"].get(name, default)
        assert entry == {
            "bits": theirs["bits"],
            "group_size": theirs["group_size"],
            "mode": "toy",
        }
        # The format's tensors, and nothing of the affine's.
        w = weights[f"{name.removeprefix('language_model.')}.weight"]
        codes, scales = tensors[f"{name}.weight"], tensors[f"{name}.scales"]
        assert (codes.shape, codes.dtype) == (w.shape, mx.uint8)
        assert (scales.shape, scales.dtype) == (w.shape[:-1], mx.float32)
        assert f"{name}.biases" not in tensors
        # They keep this layer's weights, to the half scale of the codec.
        kept = {"weight": codes, "scales": scales}
        decoded = toy.decode(kept, entry["bits"], entry["group_size"])
        error = mx.abs(decoded - w.astype(mx.float32))
        assert mx.all(error <= 0.5001 * mx.expand_dims(scales, -1))
    assert {entry["bits"] for entry in entries.values()} == {4, 5, 6, 8}
    # The format was given the source's own floats, and an importance for
    # each input: one for a matrix, one for each of a stack's.
    assert len(toy.encoded) == len(entries)
    offers = []
    for w, importance, bits, group_size in toy.encoded:
        assert w.dtype == mx.bfloat16 and importance.dtype == mx.float32
        assert importance.shape == (*w.shape[:-2], w.shape[-1])
        offers.append((w.shape, bits, group_size))
    assert sorted(offers) == sorted(
        (tensors[f"{name}.weight"].shape, entry["bits"], entry["group_size"])
        for name, entry in entries.items()
    )
    # The checkpoint names what reads it, and everything else is as without.
    assert config[_RECORD] == {
        "toy": {"plugin": "omlx-toy 0.3.1", "contract": 1, "record": {}}
    }
    said = f"oQ4: {len(entries)} layers in quant format toy"
    assert said in caplog.messages and _quoted(said)
    _assert_same_output(_beside(tensors, config, entries), _beside(*plain[:2], entries))

    # It loads as the format's layers, and computes.
    maybe_apply_pre_load_patches(str(output))
    model, _ = load_model(output, strict=True)
    layers = {
        name: layer
        for name, layer in model.named_modules()
        if isinstance(layer, FormatLinear | FormatSwitchLinear)
    }
    assert sorted(layers) == applied
    for name, layer in layers.items():
        kind = FormatSwitchLinear if name in stacks else FormatLinear
        assert type(layer) is kind and layer.mode == "toy"
        assert (layer.bits, layer.group_size) == (
            entries[name]["bits"],
            entries[name]["group_size"],
        )
        assert mx.array_equal(layer["weight"], tensors[f"{name}.weight"])
        assert mx.array_equal(layer["scales"], tensors[f"{name}.scales"])
    logits = model(mx.array([[1, 2, 3, 4]]))
    assert logits.shape == (1, 4, 64) and mx.isfinite(logits).all()


def test_what_a_format_declines_is_written_as_without_it(source, plain, toy, tmp_path):
    # A format registered by hand, which keeps the stacks alone.
    toy.answer = lambda tensors, w: tensors if w.ndim == 3 else None
    tensors, config, applied = _quantize(
        source, tmp_path / "oqe-stacks", quant_format="toy"
    )
    entries = _in_format(config)
    assert len(entries) == 6 and len(toy.encoded) == len(applied) > 6
    assert all(tensors[f"{name}.weight"].dtype == mx.uint8 for name in entries)
    assert config[_RECORD] == {"toy": {"plugin": None, "contract": 1, "record": {}}}
    _assert_same_output(_beside(tensors, config, entries), _beside(*plain[:2], entries))


def test_a_layer_keeps_the_group_size_oq_gives_it(source, toy, tmp_path):
    # oQ gives a shared expert its groups of 64 at any group size.
    _, theirs, _ = _quantize(source, tmp_path / "oqe", group_size=32)
    _, config, _ = _quantize(
        source, tmp_path / "oqe-toy", group_size=32, quant_format="toy"
    )
    sizes = {name: entry["group_size"] for name, entry in _in_format(config).items()}
    assert set(sizes.values()) == {32, 64}
    default = theirs["quantization"]
    assert sizes == {name: default.get(name, default)["group_size"] for name in sizes}
    assert sorted(group_size for *_, group_size in toy.encoded) == sorted(
        sizes.values()
    )


@pytest.mark.parametrize("declines", ["layout", "encode"])
def test_a_format_that_takes_no_layer_is_said_so(
    source, plain, toy, tmp_path, caplog, declines
):
    if declines == "layout":
        toy.lays_out = False
    else:
        toy.answer = lambda tensors, w: None
    caplog.set_level(logging.INFO, logger="omlx.oq")
    tensors, config, applied = _quantize(
        source, tmp_path / "oqe-none", quant_format="toy"
    )
    assert len(toy.offered) == len(applied)
    assert len(toy.encoded) == (0 if declines == "layout" else len(applied))
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert warnings == [
        "oQ4: quant format toy took no layer: the output is as without it"
    ]
    assert _quoted(warnings[0])
    assert not any("layers in quant format" in message for message in caplog.messages)
    assert _RECORD not in config
    _assert_same_output((tensors, config), plain[:2])


def test_without_a_format_oq_says_nothing_of_one(source, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="omlx.oq")
    _, config, _ = _quantize(source, tmp_path / "oqe")
    assert _RECORD not in config
    assert not any("quant format" in message for message in caplog.messages)


def test_a_quant_format_needs_oqe_and_a_registered_mode(source, toy, tmp_path):
    output = tmp_path / "refused"
    with pytest.raises(ValueError) as refusal:
        oq.quantize_oq_streaming(str(source), str(output), 4, quant_format="toy")
    assert str(refusal.value) == (
        "quant format 'toy' encodes with oQe's importance: pass enhanced=True"
    )
    assert _quoted(str(refusal.value))
    with pytest.raises(ValueError) as refusal:
        oq.quantize_oq_streaming(
            str(source), str(output), 4, enhanced=True, quant_format="lloyd"
        )
    assert str(refusal.value) == (
        "no quant format is registered for mode 'lloyd': install its plugin"
    )
    assert _quoted(str(refusal.value), "lloyd")
    assert not output.exists() and toy.offered == []


@pytest.mark.parametrize("model_type", ["deepseek_v41", "muse_spark"])
def test_a_model_with_a_writer_of_its_own_takes_no_format(toy, tmp_path, model_type):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": model_type}))
    output = tmp_path / "refused"
    with pytest.raises(ValueError) as refusal:
        oq.quantize_oq_streaming(
            str(source), str(output), 4, enhanced=True, quant_format="toy"
        )
    assert str(refusal.value) == (
        f"oQ writes a {model_type} checkpoint through a writer of its own, "
        "which takes no quant format"
    )
    # The page quotes it of one of the two.
    assert str(refusal.value).replace(model_type, "deepseek_v41") in _QUOTED
    assert not output.exists()


def test_a_writer_of_its_own_writes_without_a_format(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
    wrote = []
    monkeypatch.setattr(
        "omlx.patches.deepseek_v41.oq.quantize",
        lambda source, output, **kwargs: wrote.append(str(output)),
    )
    output = tmp_path / "written"
    oq.quantize_oq_streaming(str(source), str(output), 4, enhanced=True)
    assert wrote == [str(output)]


@pytest.fixture
def own_switch(monkeypatch):
    """Qwen3.5's experts in a switch layer of the family's own, as DeepSeek
    V4's and GLM's are: a ``to_quantized`` of its own, which builds MLX's
    modes and no format's layer."""

    class SwitchLinear(lm_switch.SwitchLinear):
        def to_quantized(self, group_size=64, bits=4, mode="affine"):
            built = inspect.unwrap(lm_switch.SwitchLinear.to_quantized)
            return built(self, group_size, bits, mode)

    class SwitchGLU(lm_switch.SwitchGLU):
        def __init__(self, input_dims, hidden_dims, num_experts, **kwargs):
            super().__init__(input_dims, hidden_dims, num_experts, **kwargs)
            bias = "bias" in self.gate_proj
            self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias)
            self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias)
            self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias)

    monkeypatch.setattr(qwen3_next, "SwitchGLU", SwitchGLU)
    return f"{SwitchLinear.__module__}.{SwitchLinear.__qualname__}"


def test_a_layer_whose_class_loads_no_format_stays_affine(
    source, plain, toy, own_switch, tmp_path, caplog
):
    caplog.set_level(logging.INFO, logger="omlx.oq")
    output = tmp_path / "oqe-own"
    tensors, config, applied = _quantize(source, output, quant_format="toy")
    entries = _in_format(config)
    # The stacks of experts are the family's own layers: oQ's affine. The
    # linears beside them, a shared expert's among them, are the format's.
    stacks = [name for name in applied if ".switch_mlp." in name]
    assert len(stacks) == 6
    assert sorted(entries) == sorted(set(applied) - set(stacks))
    assert any(".shared_expert." in name for name in entries)
    assert all(tensors[f"{name}.weight"].dtype == mx.uint32 for name in stacks)
    assert len(toy.encoded) == len(entries)
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    said = (
        "oQ4: 6 matrices stay oQ's affine: a quant format's layer is not "
        f"built for their class ({own_switch})"
    )
    assert warnings == [said]
    v4 = "omlx.patches.deepseek_v4.switch_layers.SwitchLinear"
    assert _quoted(said.replace(own_switch, v4))
    _assert_same_output(_beside(tensors, config, entries), _beside(*plain[:2], entries))
    # So the checkpoint loads, through the family's own to_quantized.
    maybe_apply_pre_load_patches(str(output))
    model, _ = load_model(output, strict=True)
    kinds = {type(layer).__name__ for name, layer in model.named_modules()}
    assert {"FormatLinear", "QuantizedSwitchLinear"} <= kinds
    assert "FormatSwitchLinear" not in kinds
    logits = model(mx.array([[1, 2, 3, 4]]))
    assert logits.shape == (1, 4, 64) and mx.isfinite(logits).all()


_MEASURED = (".in_sum2", ".counts")  # what an imatrix cache kept of an entry


def _without_classes(path):
    """The imatrix cache at ``path`` as one written before each entry kept
    the class of its layer."""
    with np.load(path) as data:
        kept = {name: data[name] for name in data.files}
    np.savez_compressed(
        path, **{name: a for name, a in kept.items() if name.endswith(_MEASURED)}
    )


def test_a_cache_without_the_layers_classes_is_collected_again_for_a_format(
    source, toy, tmp_path
):
    cache = tmp_path / "imatrix.npz"

    def reused(name, **kwargs):
        output = tmp_path / name
        _quantize(source, output, imatrix_cache_path=str(cache), **kwargs)
        report = json.loads((output / "oq_imatrix_report.json").read_text())
        return report["cache_reused"]

    assert not reused("first")
    _without_classes(cache)
    # A run without a format reuses it; a run with one collects again, and
    # keeps the class of each entry's layer.
    assert reused("plain")
    assert not reused("toy", quant_format="toy")
    entries = oq._load_oqe_imatrix(cache).entries
    assert all(type(entry.module_class) is str for entry in entries.values())
    assert {name: entry.module_class for name, entry in entries.items()} == {
        name: (
            "mlx_lm.models.switch_layers.SwitchLinear"
            if ".switch_mlp." in name
            else "mlx.nn.layers.linear.Linear"
        )
        for name in entries
    }
    assert reused("toy again", quant_format="toy")


_GLM5_NEXT = (
    "its linear attention fuses its input projections by hand, as MLX's affine "
    "layers or as floats"
)


@pytest.mark.parametrize(
    "family, config, why",
    [
        ("glm5_next", {"model_type": "glm5_next"}, _GLM5_NEXT),
        (
            "glm5_next_text",
            {"model_type": "glm5", "text_config": {"model_type": "glm5_next_text"}},
            _GLM5_NEXT,
        ),
        (
            "clef",
            # A Clef checkpoint is a Qwen3.5 one with Clef's joint head beside it.
            {"model_type": "qwen3_5"},
            "it reads rows of its output head by hand, as MLX's quantized "
            "layers or as floats",
        ),
    ],
    ids=["model type", "text model type", "decision family"],
)
def test_a_family_that_reads_a_format_s_layer_wrongly_is_refused(
    toy, tmp_path, family, config, why
):
    source = tmp_path / "source"
    source.mkdir()
    if family == "clef":
        for name in ("joint_head.safetensors", "joint_head_config.json"):
            (source / name).write_text("{}")
    (source / "config.json").write_text(json.dumps(config))
    output = tmp_path / "refused"
    with pytest.raises(ValueError) as refusal:
        oq.quantize_oq_streaming(
            str(source), str(output), 4, enhanced=True, quant_format="toy"
        )
    assert str(refusal.value) == (
        f"oQ writes no {family} checkpoint in quant format 'toy': {why}"
    )
    assert _quoted(str(refusal.value).replace("glm5_next_text", "glm5_next"))
    assert not output.exists() and toy.offered == []


def test_a_checkpoint_keeps_the_record_its_format_gives(
    source, toy, tmp_path, monkeypatch
):
    record = {"layout": 2, "levels": [-1.5, 0.25, 3], "name": "lloyd"}
    monkeypatch.setattr(toy, "record", lambda: dict(record))
    output = tmp_path / "oqe-toy"
    _, config, _ = _quantize(source, output, quant_format="toy")
    assert config[_RECORD] == {"toy": {"plugin": None, "contract": 1, "record": record}}
    # The load hands it back to the format, once.
    assert toy.accepted == []
    maybe_apply_pre_load_patches(str(output))
    assert toy.accepted == [record]


_CYCLE: dict = {}
_CYCLE["self"] = _CYCLE


@pytest.mark.parametrize(
    "record",
    [
        ["layout", 2],
        {"levels": (1, 2)},
        {2: "layout"},
        {"levels": mx.ones((2,))},
        {"scale": float("nan")},
        _CYCLE,
    ],
    ids=["no dict", "a tuple", "a key not a string", "an array", "a nan", "a cycle"],
)
def test_a_record_that_json_does_not_keep_is_refused(
    source, toy, tmp_path, monkeypatch, record
):
    monkeypatch.setattr(toy, "record", lambda: record)
    output = tmp_path / "refused"
    with pytest.raises(ValueError) as refusal:
        oq.quantize_oq_streaming(
            str(source), str(output), 4, enhanced=True, quant_format="toy"
        )
    assert str(refusal.value) == (
        f"quant format 'toy' gives the record {record!r}, not a dict that JSON "
        "keeps as it is: report it to the plugin"
    )
    if record == {"levels": (1, 2)}:
        assert _quoted(str(refusal.value))
    assert not output.exists() and toy.offered == []
