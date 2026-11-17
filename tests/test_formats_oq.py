# SPDX-License-Identifier: Apache-2.0
"""oQ writes a checkpoint in a quant format (``quant_format``): what a format
is offered and what is checked of its answer, then a tiny checkpoint written
in a toy format and loaded back.
"""

from __future__ import annotations

import inspect
import json
import logging
import sys
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx_lm.utils
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import llama, qwen3_5_moe, qwen3_next
from mlx_lm.models import switch_layers as lm_switch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from omlx import formats, oq
from omlx.formats.linear import FormatLinear, FormatSwitchLinear
from omlx.patches.format_load import loads_in_format
from omlx.utils.model_loading import maybe_apply_pre_load_patches
from tests.test_formats_kernels import _BLOCKS, _PAGE
from tests.test_formats_linear import _ToyFormat

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

_RECORD = "omlx_quant_formats"  # in a checkpoint's config: what reads its layers
# docs/quant-formats.md quotes what oQ says of a format, one named myformat.
_QUOTED = " ".join(_PAGE.split())


def _quoted(message, name="toy"):
    return message.replace(name, "myformat") in _QUOTED


# The families verified with quant formats, as this oMLX ships them.
_VERIFIED = getattr(oq, "_QUANT_FORMAT_VERIFIED", frozenset())


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    """The toy checkpoint's family taken as verified: oMLX patches Qwen3.5
    MoE on an mlx-vlm load, and oQ writes a family it patches in a format
    only once the family is verified."""
    monkeypatch.setattr(
        oq,
        "_QUANT_FORMAT_VERIFIED",
        frozenset({"qwen3_5_moe_text"}),
        raising=False,
    )


class _Codec(_ToyFormat):
    """The toy format with its codec: a byte for each weight and a scale for
    each row, which keep a weight to half a scale. ``answer`` stands in for
    a codec that declines, or has a defect."""

    def __init__(self):
        super().__init__()
        self.offered, self.encoded, self.lays_out = [], [], True
        self.answer = lambda tensors, w: tensors
        # What the loads handed it, kept off its own state: a format that
        # changes its state to accept a checkpoint is refused.
        self.log = SimpleNamespace(accepted=[])

    @property
    def accepted(self):
        return self.log.accepted

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


def load_model(path, **kwargs):
    """mlx-lm's loader as the load patch leaves it: the function a module
    bound at its import is the library's own, which builds no layer of a
    format."""
    return mlx_lm.utils.load_model(path, **kwargs)


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
    _save_tokenizer(path, config["vocab_size"])
    return path


def _save_tokenizer(path, size):
    vocab = {f"t{i}": i for i in range(size)}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="t0"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0").save_pretrained(
        path
    )


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


def test_a_format_is_asked_of_a_matrix_whose_k_oq_s_groups_do_not_divide(
    source, toy, tmp_path
):
    # In groups of 128, oQ's affine keeps the matrices of K = 64 in floats.
    theirs = _quantize(source, tmp_path / "oqe", group_size=128)
    output = tmp_path / "oqe-toy"
    tensors, config, applied = _quantize(
        source, output, group_size=128, quant_format="toy"
    )
    entries = _in_format(config)
    floats = sorted(set(applied) - set(theirs[2]))
    assert floats and all(
        theirs[0][f"{name}.weight"].shape[-1] == 64 for name in floats
    )
    assert not any(f"{name}.scales" in theirs[0] for name in floats)
    # The format is asked of each, at the group size oQ gives it, and keeps it.
    assert set(floats) <= set(entries)
    assert {entries[name]["group_size"] for name in floats} == {128}
    assert sorted(entries) == applied
    _assert_same_output(
        _beside(tensors, config, entries), _beside(*theirs[:2], entries)
    )
    maybe_apply_pre_load_patches(str(output))
    model, _ = load_model(output, strict=True)
    layers = dict(model.named_modules())
    kinds = {type(layers[name]) for name in floats}
    assert kinds == {FormatLinear, FormatSwitchLinear}
    logits = model(mx.array([[1, 2, 3, 4]]))
    assert logits.shape == (1, 4, 64) and mx.isfinite(logits).all()


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


def test_without_a_format_oq_says_nothing_of_one(source, tmp_path, caplog, monkeypatch):
    walked = []
    monkeypatch.setattr(oq, "_module_classes", lambda *a, **k: walked.append(a))
    caplog.set_level(logging.INFO, logger="omlx.oq")
    _, config, _ = _quantize(source, tmp_path / "oqe")
    assert _RECORD not in config
    assert not any("quant format" in message for message in caplog.messages)
    # Nor builds a model to read the classes of its modules.
    assert walked == []


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


@pytest.mark.parametrize("model_type", ["deepseek_v41", "muse_spark", "Muse-Spark"])
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
        "which takes no quant format; write it without a quant format"
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


def own_switch(monkeypatch):
    """Qwen3.5's experts in a switch layer of the family's own, as DeepSeek
    V4's and GLM's are: a ``to_quantized`` of its own, which builds MLX's
    modes and no format's layer. Returns the class's name."""

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
    source, plain, toy, tmp_path, monkeypatch, caplog
):
    own = own_switch(monkeypatch)
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
        f"built for their class ({own})"
    )
    assert warnings == [said]
    v4 = "omlx.patches.deepseek_v4.switch_layers.SwitchLinear"
    assert _quoted(said.replace(own, v4))
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
    """The imatrix cache at ``path`` as upstream writes one: each entry's
    sums and counts, nothing else."""
    with np.load(path) as data:
        kept = {name: data[name] for name in data.files}
    np.savez_compressed(
        path, **{name: a for name, a in kept.items() if name.endswith(_MEASURED)}
    )


def _cached(source, tmp_path, cache, name, **kwargs):
    """A run of ``_quantize`` over the imatrix cache at ``cache``: its
    output's layers in the toy format, and whether it reused the cache."""
    output = tmp_path / name
    _, config, _ = _quantize(source, output, imatrix_cache_path=str(cache), **kwargs)
    report = json.loads((output / "oq_imatrix_report.json").read_text())
    return sorted(_in_format(config)), report["cache_reused"]


def test_a_cache_written_without_classes_serves_a_run_with_a_format(
    source, toy, tmp_path, monkeypatch
):
    fresh, reused = _cached(
        source, tmp_path, tmp_path / "fresh.npz", "fresh", quant_format="toy"
    )
    assert fresh and not reused
    cache = tmp_path / "imatrix.npz"
    assert _cached(source, tmp_path, cache, "plain") == ([], False)
    _without_classes(cache)
    # No calibration pass, and the matrices of a fresh collection.
    calibrated = []
    monkeypatch.setattr(oq, "_collect_imatrix", lambda *a, **k: calibrated.append(a))
    assert _cached(source, tmp_path, cache, "toy", quant_format="toy") == (fresh, True)
    assert calibrated == []


def test_each_run_reads_the_classes_of_the_model_it_writes(
    source, toy, tmp_path, monkeypatch
):
    # A cache collected while the experts were mlx-lm's switch layers serves
    # a run where they are a switch layer of the family's own: they stay
    # affine, and the checkpoint loads.
    cache = tmp_path / "imatrix.npz"
    stock, reused = _cached(source, tmp_path, cache, "stock", quant_format="toy")
    assert any(".switch_mlp." in name for name in stock) and not reused
    own_switch(monkeypatch)
    own, reused = _cached(source, tmp_path, cache, "own", quant_format="toy")
    assert reused and own == [name for name in stock if ".switch_mlp." not in name]
    maybe_apply_pre_load_patches(str(tmp_path / "own"))
    model, _ = load_model(tmp_path / "own", strict=True)
    assert "FormatSwitchLinear" not in {
        type(m).__name__ for _, m in model.named_modules()
    }


_GLM5_NEXT = (
    "its linear attention fuses its input projections by hand, as MLX's affine "
    "layers or as floats"
)


_DEEPSEEK_V4 = "its DSpark draft reads the output head's weight by hand, as floats"
_NOT_VERIFIED = (
    "it is no family served by mlx-lm's own code that oMLX proves it does not "
    "patch, nor one verified with quant formats end to end"
)
# The page quotes these of the same family, or of one of their kind.
_PAGE_FAMILY = {"glm5_next_text": "glm5_next", "deepseek_v4_mtp": "deepseek_v4"}
# The pages quote the refusals of these families.
_QUOTED_FAMILIES = {
    "glm5_next",
    "glm5_next_text",
    "raon1_mini",
    "deepseek_v4",
    "deepseek_v4_mtp",
    "glm_moe_dsa",
    "step3p7",
    "clef",
    "inkling2",
}


@pytest.mark.parametrize(
    "family, config, why",
    [
        ("glm5_next", {"model_type": "glm5_next"}, _GLM5_NEXT),
        (
            "glm5_next_text",
            {"model_type": "glm5", "text_config": {"model_type": "GLM5-Next-Text"}},
            _GLM5_NEXT,
        ),
        (
            "raon1_mini",
            {"model_type": "raon1_mini"},
            "its merged gate and up projection is sliced by hand on a prefill, as "
            "MLX's quantized layer or as floats",
        ),
        ("deepseek_v4", {"model_type": "deepseek_v4"}, _DEEPSEEK_V4),
        ("deepseek_v4_mtp", {"model_type": "deepseek_v4_mtp"}, _DEEPSEEK_V4),
        (
            "glm_moe_dsa",
            {"model_type": "glm_moe_dsa"},
            "its indexer fuses its projections' 8-bit affine layers by hand, and "
            "refuses a load where one of them is a format's",
        ),
        ("step3p7", {"model_type": "step3p7"}, _NOT_VERIFIED),
        # Under any spelling.
        ("step3p7", {"model_type": "STEP3P7"}, _NOT_VERIFIED),
        ("unlimited_ocr", {"model_type": "unlimited-ocr"}, _NOT_VERIFIED),
        ("unlimited_ocr", {"model_type": "Unlimited_OCR"}, _NOT_VERIFIED),
        ("qwen3_5_text", {"model_type": "qwen3_5_text"}, _NOT_VERIFIED),
        # Families the static reader of 57e0cbf missed: they are no stock
        # family, so they are refused.
        ("minicpmv4_7", {"model_type": "minicpmv4_7"}, _NOT_VERIFIED),
        ("olmoe", {"model_type": "olmoe"}, _NOT_VERIFIED),
        ("qwen3_omni_moe", {"model_type": "qwen3_omni_moe"}, _NOT_VERIFIED),
        ("xlm_roberta", {"model_type": "xlm-roberta"}, _NOT_VERIFIED),
        # One nobody has heard of: the gate fails closed.
        ("brand_new", {"model_type": "brand-new"}, _NOT_VERIFIED),
        # A stock top level with a text model that is not.
        (
            "gemma4_text",
            {"model_type": "llama", "text_config": {"model_type": "gemma4_text"}},
            _NOT_VERIFIED,
        ),
        (
            "d1",
            # Its backbone is not stock either; it is refused as its decision
            # family.
            {"model_type": "lfm3_vl", "architectures": ["D1Model"]},
            _NOT_VERIFIED,
        ),
        (
            "clef",
            # A Clef checkpoint is a Qwen3.5 one with Clef's joint head beside it.
            {"model_type": "qwen3_5"},
            "it reads rows of its output head by hand, as MLX's quantized "
            "layers or as floats",
        ),
        (
            "inkling2",
            {"model_type": "inkling2"},
            "its DSpark draft folds the output head into its own projection by "
            "hand, as MLX's quantized layer or as floats",
        ),
    ],
    ids=[
        "model type",
        "text model type",
        "raon1_mini",
        "deepseek_v4",
        "deepseek_v4_mtp",
        "glm_moe_dsa",
        "patched",
        "spelled in capitals",
        "spelled with a hyphen",
        "spelled with capitals and an underscore",
        "renamed by the dispatcher",
        "minicpmv4_7",
        "olmoe",
        "qwen3_omni_moe",
        "xlm-roberta",
        "unknown",
        "text model not stock",
        "decision model",
        "decision family",
        "inkling2",
    ],
)
def test_a_family_neither_stock_nor_verified_is_refused(
    toy, tmp_path, family, config, why
):
    source = tmp_path / "source"
    source.mkdir()
    if family == "clef":
        for name in ("joint_head.safetensors", "joint_head_config.json"):
            (source / name).write_text("{}")
    (source / "config.json").write_text(json.dumps(config))
    output = tmp_path / "refused"
    with pytest.raises(formats.FormatRefusedError) as refusal:
        oq.quantize_oq_streaming(
            str(source), str(output), 4, enhanced=True, quant_format="toy"
        )
    assert str(refusal.value) == (
        f"quant format 'toy': oQ writes no {family} checkpoint in it: {why}; "
        "write it without a quant format, or have the family verified"
    )
    assert refusal.value.mode == "toy" and refusal.value.path is None
    # The page quotes it, of the same family or one of its kind.
    said = str(refusal.value)
    for spelled, quoted in _PAGE_FAMILY.items():
        said = said.replace(spelled, quoted)
    quoted_spelling = config["model_type"] in (family, "glm5", "qwen3_5")
    if family in _QUOTED_FAMILIES and quoted_spelling:
        assert _quoted(said)
    assert not output.exists() and toy.offered == []


def test_the_families_oq_writes_in_a_format(tmp_path, monkeypatch):
    stock = {"llama", "mistral", "qwen2", "phi3", "starcoder2", "olmo2"}
    assert set(oq._QUANT_FORMAT_STOCK) == stock
    # None is verified yet: Qwen3.8 Flash-Next enters with the run that
    # verifies it.
    assert not _VERIFIED
    source = tmp_path / "source"
    source.mkdir()
    for model_type in ("llama", "Mistral", "qwen2", "PHI3", "starcoder2", "olmo2"):
        config = {"model_type": model_type}
        assert oq._unverified_family(source, config) is None
    # A text model of a stock family under a stock top level.
    config = {"model_type": "llama", "text_config": {"model_type": "mistral"}}
    assert oq._unverified_family(source, config) is None
    # A verified family, top level or text model.
    monkeypatch.setattr(oq, "_QUANT_FORMAT_VERIFIED", frozenset({"qwen4_exp"}))
    assert oq._unverified_family(source, {"model_type": "Qwen4-Exp"}) is None
    nested = {"model_type": "llama", "text_config": {"model_type": "qwen4_exp"}}
    assert oq._unverified_family(source, nested) is None
    assert oq._unverified_family(source, {"model_type": "qwen3"}) == (
        "qwen3",
        _NOT_VERIFIED,
    )


# What every family oMLX serves runs of oMLX's patches, whichever it is:
# the entry points applied at every load and the model-independent routes a
# forward takes. A family's own patch is none of them.
_EVERY_LOAD = {
    "arrays_cache_extract.py:apply_arrays_cache_extract_guard",
    "arrays_cache_metadata.py:apply_arrays_cache_metadata_patch",
    "batched_decode_qmm.py:apply_batched_decode_qmm_patch",
    "format_load.py:apply_format_load_patch",
    "m5_gather_qmm.py:apply_m5_gather_qmm_workaround",
    # These walk any model the engine loads, and fuse MoE switch layers only.
    "moe_gate_up_fusion.py:apply_moe_gate_up_fusion",
    "moe_gate_up_fusion.py:apply_switch_glu_gate_up_fusion",
    "qwen35_moe_gate_up.py:apply_qwen35_moe_gate_up_fusion",
    "qwen35_fa256_attention.py:apply_qwen35_fa256_attention_patch",
    "qwen35_moe_router.py:apply_qwen35_moe_router_patch",
    "qwen35_moe_weighted_sum.py:apply_qwen35_moe_weighted_sum_patch",
    "qwen35_q4_mlp.py:apply_qwen35_q4_lm_prefill_linear_patch",
    "qwen35_q4_mlp.py:apply_qwen35_q4_mlp_patch",
    "qwen35_ragged_decode.py:apply_qwen35_ragged_decode_patch",
    "qwen35_yarn.py:apply_qwen35_yarn_patch",
    "sdpa256_attention.py:apply_sdpa256_attention_patch",
}
_EVERY_FORWARD = {
    "sdpa256_attention.py:_over_vector_thread_limit",
    "sdpa256_attention.py:_should_route",
    "sdpa256_attention.py:patched_sdpa",
}
# The model type each stock family is served as, and what its tiny config
# needs beside the common one.
_STOCK = {
    "llama": ("llama", {}),
    "mistral": ("llama", {}),
    "qwen2": ("qwen2", {}),
    "phi3": ("phi3", {}),
    "starcoder2": ("starcoder2", {}),
    "olmo2": ("olmo2", {}),
}


# Run in a process of its own, as a server that starts and serves the
# checkpoint: what another test applied (a family's loader patched into the
# process) is not the family's.
_SERVE = textwrap.dedent("""
    import asyncio, json, sys, threading
    from pathlib import Path
    import mlx.core as mx
    import omlx
    from omlx.engine.batched import BatchedEngine

    patches = str(Path(omlx.__file__).parent / "patches")
    loads, forwards = set(), set()
    seen = {"phase": loads}

    def record(frame, event, arg):
        code = frame.f_code
        if seen["phase"] is forwards and (
            threading.current_thread() is not threading.main_thread()
        ):
            # The engine's own threads, not the model's forward.
            return
        if event == "call" and code.co_filename.startswith(patches):
            where = f"{code.co_filename[len(patches) + 1:]}:{code.co_name}"
            if seen["phase"] is forwards or code.co_name.startswith(
                ("apply_", "configure_", "install_")
            ):
                seen["phase"].add(where)

    engine = BatchedEngine(sys.argv[1])
    # This thread's, and the threads the engine loads on.
    sys.setprofile(record)
    threading.setprofile(record)
    asyncio.run(engine.start())
    seen["phase"] = forwards
    mx.eval(engine._model(mx.array([[1, 2, 3]])))
    sys.setprofile(None)
    asyncio.run(engine.stop())
    print(json.dumps({"loads": sorted(loads), "forwards": sorted(forwards)}))
""")


@pytest.mark.parametrize("family", sorted(_STOCK))
def test_omlx_patches_no_stock_family(tmp_path, family):
    import importlib
    import subprocess
    import sys

    from omlx.model_discovery import detect_model_type

    assert set(_STOCK) == oq._QUANT_FORMAT_STOCK
    module_name, extra = _STOCK[family]
    config = {
        "model_type": family,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "vocab_size": 64,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        **extra,
    }
    module = importlib.import_module(f"mlx_lm.models.{module_name}")
    model = module.Model(module.ModelArgs.from_dict(config))
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(config))
    _save_tokenizer(tmp_path, config["vocab_size"])
    # oMLX serves it as an LLM, by its engine's own start.
    assert detect_model_type(tmp_path) == "llm"
    served = subprocess.run(
        [sys.executable, "-c", _SERVE, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(oq.__file__).parents[1],
    )
    ran = json.loads(served.stdout.strip().splitlines()[-1])
    # No patch of oMLX's but those every family gets ran for it.
    loads, forwards = set(ran["loads"]), set(ran["forwards"])
    assert loads <= _EVERY_LOAD, sorted(loads - _EVERY_LOAD)
    assert forwards <= _EVERY_FORWARD, sorted(forwards - _EVERY_FORWARD)
    assert "format_load.py:apply_format_load_patch" in loads


def test_a_verified_decision_family_is_written_whatever_its_backbone(
    tmp_path, monkeypatch
):
    source = tmp_path / "d1"
    source.mkdir()
    config = {"model_type": "lfm3_vl", "architectures": ["D1Model"]}
    (source / "config.json").write_text(json.dumps(config))
    assert oq._unverified_family(source, config)[0] == "d1"
    monkeypatch.setattr(oq, "_QUANT_FORMAT_VERIFIED", frozenset({"d1"}))
    assert oq._unverified_family(source, config) is None


@pytest.fixture(scope="module")
def llama_source(tmp_path_factory):
    """A tiny Llama checkpoint in bfloat16: a model oMLX serves with mlx-lm's
    own code."""
    config = {
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "vocab_size": 64,
        "tie_word_embeddings": False,
    }
    mx.random.seed(3)
    model = llama.Model(llama.ModelArgs.from_dict(config))
    model.set_dtype(mx.bfloat16)
    path = tmp_path_factory.mktemp("llama")
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    (path / "config.json").write_text(json.dumps(config))
    _save_tokenizer(path, config["vocab_size"])
    return path


def test_a_model_omlx_does_not_patch_is_written_in_a_format(
    llama_source, toy, tmp_path, monkeypatch
):
    monkeypatch.setattr(oq, "_QUANT_FORMAT_VERIFIED", _VERIFIED, raising=False)
    output = tmp_path / "oqe-toy"
    _, config, applied = _quantize(llama_source, output, quant_format="toy")
    # Each layer's seven linears.
    assert applied == sorted(
        f"model.layers.{i}.{name}"
        for i in range(2)
        for name in (
            "mlp.down_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "self_attn.k_proj",
            "self_attn.o_proj",
            "self_attn.q_proj",
            "self_attn.v_proj",
        )
    )
    assert sorted(_in_format(config)) == applied
    maybe_apply_pre_load_patches(str(output))
    model, _ = load_model(output, strict=True)
    layers = dict(model.named_modules())
    assert {type(layers[name]) for name in applied} == {FormatLinear}
    logits = model(mx.array([[1, 2, 3, 4]]))
    assert logits.shape == (1, 4, 64) and mx.isfinite(logits).all()


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
    model, _ = load_model(output, strict=True)
    assert toy.accepted == [record]
    layers = [m for _, m in model.named_modules() if isinstance(m, FormatLinear)]
    assert layers and all(layer.fmt is toy for layer in layers)


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


def test_an_importance_may_be_the_entry_of_another_layer():
    # GLM-5.3's indexer weighs weights_proj by wk's inputs: the same entry.
    wk, other = (oq.OQImatrixEntry(np.ones(4), np.ones(1)) for _ in range(2))
    indexer = "l.self_attn.indexer."
    entries = {f"{indexer}wk": wk, "l.mlp": other}
    imatrix = oq.OQImatrixData(entries, {}, "cache")
    config = {"model_type": "glm5_next"}
    assert oq._imatrix_entry(imatrix, f"{indexer}weights_proj", config) is wk
    assert oq._imatrix_entry(imatrix, "l.mlp", config) is other
    assert oq._imatrix_entry(imatrix, f"{indexer}weights_proj", {}) is None
    # The text model's type says so too; a weights_proj of its own, or of
    # no indexer, keeps its own entry or none.
    text = {"model_type": "glm5", "text_config": {"model_type": "glm5_next_text"}}
    assert oq._imatrix_entry(imatrix, f"{indexer}weights_proj", text) is wk
    own = oq.OQImatrixEntry(np.ones(4), np.ones(1))
    imatrix.entries[f"{indexer}weights_proj"] = own
    assert oq._imatrix_entry(imatrix, f"{indexer}weights_proj", config) is own
    imatrix.entries["l.mlp.wk"] = other
    assert oq._imatrix_entry(imatrix, "l.mlp.weights_proj", config) is None
    del imatrix.entries[f"{indexer}weights_proj"]
    importance = oq._lookup_imatrix_importance(
        imatrix,
        f"{indexer}weights_proj.weight",
        (8, 4),
        config=config,
        strict=True,
        report=None,
    )
    assert np.array_equal(np.array(importance), np.ones(4))


def test_a_matrix_of_no_module_of_its_path_stays_affine_and_is_named(
    source, toy, tmp_path, monkeypatch, caplog
):
    walk = oq._module_classes
    lost = "language_model.model.layers.1.mlp.shared_expert.down_proj"

    def without(*args, **kwargs):
        return {
            name: cls for name, cls in walk(*args, **kwargs).items() if name != lost
        }

    monkeypatch.setattr(oq, "_module_classes", without)
    caplog.set_level(logging.INFO, logger="omlx.oq")
    _, config, applied = _quantize(source, tmp_path / "oqe-toy", quant_format="toy")
    assert lost in applied and sorted(_in_format(config)) == sorted(
        set(applied) - {lost}
    )
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    said = (
        "oQ4: 1 matrices stay oQ's affine: a quant format's layer is not built for "
        "their class (no module holds them)\npaths that no module of the model it "
        f"loads as holds at their shape: {lost}"
    )
    assert warnings == [said]
    # The page quotes the line that names them.
    named = said.splitlines()[1].replace(lost, "model.layers.0.mlp.experts.0.gate_proj")
    assert _quoted(named)


def test_the_warning_names_twenty_paths_and_counts_the_others():
    cls = "omlx.patches.deepseek_v4.switch_layers.SwitchLinear"
    paths = [f"m.layers.{i:02d}.mlp.gate_proj" for i in range(23)]
    said = oq._left_affine("3", [cls, cls], paths[::-1])
    head = (
        "oQ3: 25 matrices stay oQ's affine: a quant format's layer is not built for "
        f"their class (no module holds them, {cls})\npaths that no module of the "
        "model it loads as holds at their shape: "
    )
    assert said == head + ", ".join(paths[:20]) + " and 3 more"
    assert oq._left_affine("3", [], paths[:20]) == (
        head.replace("25", "20").replace(f", {cls}", "") + ", ".join(paths[:20])
    )
    assert oq._left_affine("4", [cls], []) == (
        "oQ4: 1 matrices stay oQ's affine: a quant format's layer is not built for "
        f"their class ({cls})"
    )


def test_a_matrix_of_another_shape_than_its_module_s_stays_affine(
    source, toy, tmp_path, monkeypatch, caplog
):
    walk = oq._module_classes
    other = "language_model.model.layers.1.mlp.switch_mlp.down_proj"

    def reshaped(*args, **kwargs):
        walked = walk(*args, **kwargs)
        cls, loads, shape = walked[other]
        return {**walked, other: (cls, loads, (shape[0] + 1, *shape[1:]))}

    monkeypatch.setattr(oq, "_module_classes", reshaped)
    caplog.set_level(logging.INFO, logger="omlx.oq")
    _, config, applied = _quantize(source, tmp_path / "oqe-toy", quant_format="toy")
    assert other in applied and other not in _in_format(config)
    assert sorted(_in_format(config)) == sorted(set(applied) - {other})
    assert f"at their shape: {other}" in caplog.text


def test_the_walk_builds_the_written_model_from_its_config_alone(
    source, tmp_path, monkeypatch
):
    from omlx.patches import format_load

    asked = []

    def loads(cls):
        asked.append(cls)
        return loads_in_format(cls)

    monkeypatch.setattr(format_load, "loads_in_format", loads)
    config = json.loads((source / "config.json").read_text())
    walked = oq._module_classes(str(source), config)
    # Whether a format's layer is built for a class is asked once.
    assert sorted(asked) == sorted({cls for cls, _, _ in walked.values()})
    # Each module of the model, its class, whether a format's layer is built
    # for it, and its weight's shape.
    gate = "language_model.model.layers.0.mlp.switch_mlp.gate_proj"
    lm_switch_linear = "mlx_lm.models.switch_layers.SwitchLinear"
    assert walked[gate] == (lm_switch_linear, True, (4, 128, 64))
    linear = walked["language_model.model.layers.0.mlp.shared_expert.up_proj"]
    assert linear == ("mlx.nn.layers.linear.Linear", True, (128, 64))
    norm = walked["language_model.model.norm"]
    assert norm[1] is False and norm[2] == (64,)
    # From the config it is given, unquantized, reading neither a weight nor
    # the tokenizer: a copy of the source with its weights and tokenizer
    # broken walks the same.
    broken = tmp_path / "broken"
    broken.mkdir()
    for item in source.iterdir():
        if item.suffix == ".json" and "token" not in item.name:
            (broken / item.name).write_bytes(item.read_bytes())
    (broken / "model.safetensors").write_bytes(b"not safetensors")
    (broken / "tokenizer.json").write_text("{")
    assert oq._module_classes(str(broken), config) == walked
    assert (broken / "model.safetensors").read_bytes() == b"not safetensors"
    # A run's quantization names no layer: none is quantized.
    quantized = {**config, "quantization": {"group_size": 64, "bits": 4}}
    assert oq._module_classes(str(broken), quantized) == walked
    one = oq._module_classes(str(broken), {**config, "num_hidden_layers": 1})
    assert not any(".layers.1." in name for name in one)


def test_the_walk_keeps_the_process_s_load_state(source, monkeypatch):
    from omlx.patches import mlx_lm_mtp, mlx_vlm_mtp, qwen35_yarn
    from omlx.utils import model_loading

    # Whatever the pre-load patches set for the model the walk builds.
    patches = model_loading.maybe_apply_pre_load_patches

    def setting(*args, **kwargs):
        patches(*args, **kwargs)
        mlx_lm_mtp._MTP_ACTIVE = not mlx_lm_mtp._MTP_ACTIVE
        mlx_lm_mtp._MTP_DEPTH, mlx_lm_mtp._MTP_DEPTH_FIXED = 3, True
        mlx_vlm_mtp._MTP_ATTACH_ENABLED = False
        qwen35_yarn._context_windows[str(source)] = 1024
        qwen35_yarn._context_windows.pop("served", None)

    monkeypatch.setattr(model_loading, "maybe_apply_pre_load_patches", setting)
    monkeypatch.setitem(qwen35_yarn._context_windows, "served", 65536)
    windows = qwen35_yarn._context_windows
    before = (
        mlx_lm_mtp._MTP_ACTIVE,
        mlx_lm_mtp._MTP_DEPTH,
        mlx_lm_mtp._MTP_DEPTH_FIXED,
        mlx_vlm_mtp._MTP_ATTACH_ENABLED,
        dict(windows),
    )
    config = json.loads((source / "config.json").read_text())
    assert oq._module_classes(str(source), config)
    after = (
        mlx_lm_mtp._MTP_ACTIVE,
        mlx_lm_mtp._MTP_DEPTH,
        mlx_lm_mtp._MTP_DEPTH_FIXED,
        mlx_vlm_mtp._MTP_ATTACH_ENABLED,
        dict(qwen35_yarn._context_windows),
    )
    assert after == before and qwen35_yarn._context_windows is windows


class _Walked(nn.Module):
    """What a stubbed loader builds: one linear, and whether it was closed."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 16)
        self.closed = False

    def close(self):
        self.closed = True


def _stub_loaders(monkeypatch):
    """mlx-lm's and mlx-vlm's load_model stubbed: each call's library, thread
    and arguments, and the model each returns."""
    from mlx_vlm import utils as vlm_utils

    calls, model = [], _Walked()

    def loader(library):
        def load(path, **kwargs):
            view = Path(path)
            config = json.loads((view / "config.json").read_text())
            index = json.loads((view / "model.safetensors.index.json").read_text())
            thread = threading.current_thread().name
            calls.append((library, thread, kwargs, config, index))
            return model if library == "mlx-vlm" else (model, None)

        return load

    monkeypatch.setattr(mlx_lm.utils, "load_model", loader("mlx-lm"))
    monkeypatch.setattr(vlm_utils, "load_model", loader("mlx-vlm"))
    return calls, model


def _checkpoint(path, config, tensors=("model.embed_tokens.weight",)):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    weights = {name: mx.zeros((2, 64)) for name in tensors}
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    return path


@pytest.mark.parametrize(
    "config, library",
    [
        ({"model_type": "llama"}, "mlx-lm"),
        # mlx-vlm serves this one without a vision config, as a text-only
        # run of it writes it.
        ({"model_type": "lfm3_audio"}, "mlx-vlm"),
        ({"model_type": "qwen3_vl", "vision_config": {"depth": 1}}, "mlx-vlm"),
        # A text-only Gemma 4 is served by mlx-lm, unless it has an MTP head.
        ({"model_type": "gemma4_unified"}, "mlx-lm"),
        (
            {
                "model_type": "gemma4",
                "text_config": {"mtp_assistant_config": {"num_layers": 1}},
            },
            "mlx-vlm",
        ),
    ],
    ids=["llama", "lfm3_audio", "qwen3_vl", "gemma4_unified", "gemma4 with MTP"],
)
def test_the_walk_builds_the_model_with_the_library_that_serves_it(
    tmp_path, monkeypatch, config, library
):
    from omlx.model_discovery import _register_model

    calls, model = _stub_loaders(monkeypatch)
    source = _checkpoint(tmp_path / "source", config)
    walked = oq._module_classes(str(source), config)
    ((loaded, thread, kwargs, given, index),) = calls
    # As oMLX's model discovery registers the checkpoint.
    found = {}
    _register_model(found, source, "source")
    assert loaded == library
    assert found["source"].engine_type == ("vlm" if library == "mlx-vlm" else "batched")
    # Lazily, unchecked, on the thread the engines load models on; the
    # model is let go of.
    assert kwargs["lazy"] is True and kwargs["strict"] is False
    assert thread.startswith("mlx-global") and model.closed
    assert given == config and walked["proj"][2] == (16, 64)
    # The view: the source's files but its weights and config, and an index
    # declaring the source's tensors.
    assert index == {"weight_map": {"model.embed_tokens.weight": "model.safetensors"}}


def test_the_walk_is_of_models_the_llm_and_vlm_engines_serve(tmp_path, monkeypatch):
    calls, _ = _stub_loaders(monkeypatch)
    config = {"model_type": "bert", "architectures": ["BertModel"]}
    source = _checkpoint(tmp_path / "source", config)
    with pytest.raises(ValueError, match="with 'embedding'"):
        oq._module_classes(str(source), config)
    assert calls == []


def test_the_walk_declares_the_source_s_tensors_whatever_its_files(
    tmp_path, monkeypatch
):
    calls, _ = _stub_loaders(monkeypatch)
    config = {"model_type": "llama"}
    source = _checkpoint(tmp_path / "source", config)
    # An index that does not parse: the files' headers.
    (source / "model.safetensors.index.json").write_text("{")
    # A file of weights that is a link to nothing declares nothing.
    (source / "extra.safetensors").symlink_to(tmp_path / "nowhere")
    oq._module_classes(str(source), config)
    declared = {"weight_map": {"model.embed_tokens.weight": "model.safetensors"}}
    assert calls[-1][4] == declared
    # The source's own index is not written through.
    assert (source / "model.safetensors.index.json").read_text() == "{"
    index = {"weight_map": {"model.norm.weight": "model.safetensors"}}
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    oq._module_classes(str(source), config)
    assert calls[-1][4] == {"weight_map": {"model.norm.weight": "model.safetensors"}}
    assert json.loads((source / "model.safetensors.index.json").read_text()) == index


def test_the_calibration_patches_the_mtp_runtime_as_upstream_does(
    tmp_path, monkeypatch
):
    from omlx.patches import mlx_vlm_mtp

    asked = []
    monkeypatch.setattr(
        mlx_vlm_mtp, "apply_mlx_vlm_mtp_runtime_patch", lambda *a: asked.append(a)
    )
    monkeypatch.setattr(mlx_vlm_mtp, "apply_mlx_vlm_mtp_patch", lambda: True)
    config = {
        "model_type": "qwen3_vl",
        "vision_config": {"depth": 1},
        "mtp_num_hidden_layers": 1,
    }
    source = _checkpoint(tmp_path / "source", config, tensors=("mtp.fc.weight",))
    with oq._calibration_patches(source, config):
        pass
    # Unscoped, as upstream's calibration does (UPSTREAM-BUGS O16).
    assert asked == [()]


def test_the_engines_apply_pre_load_patches_on_the_thread_that_loads():
    import ast
    import inspect

    from omlx.engine import batched, vlm

    for engine, loader in (
        (batched.BatchedEngine, "_load_model_sync"),
        (vlm.VLMBatchedEngine, "_load_vlm_sync"),
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(engine.start)))
        (start,) = tree.body
        nested = next(
            node
            for node in ast.walk(start)
            if isinstance(node, ast.FunctionDef) and node.name == loader
        )

        def dispatches(node):
            return sum(
                isinstance(call, ast.Call)
                and getattr(call.func, "id", None) == "maybe_apply_pre_load_patches"
                for call in ast.walk(node)
            )

        # Once, inside the function the global MLX executor runs: the walk of
        # a format run, on that thread too, never sets or restores that state
        # while an engine loads.
        assert dispatches(nested) == dispatches(start) == 1, engine


def test_the_walk_reads_no_sidecar_and_keeps_the_calibration_override(
    tmp_path, monkeypatch
):
    calls, _ = _stub_loaders(monkeypatch)
    # MiMo's MTP sidecar would be read by its sanitize: the walk names none.
    mimo = {"model_type": "mimo_v2"}
    source = _checkpoint(tmp_path / "mimo", mimo)
    (source / "mtp").mkdir()
    (source / "mtp" / "model_mtp.safetensors").write_bytes(b"")
    oq._module_classes(str(source), mimo)
    assert calls[-1][0] == "mlx-lm" and calls[-1][2]["model_config"] is None
    # A QAT config's quantization is not the walk's.
    qat = {"model_type": "llama", "quantization_config": {"quant_method": "qat"}}
    monkeypatch.setattr(oq, "_is_qat_unquantized_config", lambda qc: bool(qc))
    source = _checkpoint(tmp_path / "qat", qat)
    oq._module_classes(str(source), qat)
    assert calls[-1][2]["model_config"] == {"quantization_config": None}


def test_the_walk_leaves_the_source_as_it_is(tmp_path, monkeypatch):
    from omlx.utils import model_loading

    calls, _ = _stub_loaders(monkeypatch)
    patches, dispatched = model_loading.maybe_apply_pre_load_patches, []

    def dispatching(model_name, *args, **kwargs):
        dispatched.append(json.loads((Path(model_name) / "config.json").read_text()))
        return patches(model_name, *args, **kwargs)

    monkeypatch.setattr(model_loading, "maybe_apply_pre_load_patches", dispatching)
    config = {"model_type": "llama"}
    source = _checkpoint(tmp_path / "source", config)
    before = (source / "config.json").read_bytes()
    # Links that resolve to nothing, or to themselves, as a source may hold.
    (source / "dangling").symlink_to(tmp_path / "nowhere")
    (source / "loop").symlink_to(source / "loop")
    written = {**config, "torch_dtype": "float16"}
    oq._module_classes(str(source), written)
    assert (source / "config.json").read_bytes() == before
    # The loader and the pre-load dispatch read the config the run writes.
    assert calls[0][3] == written and dispatched == [written]


def test_the_walk_runs_on_the_engines_load_thread_too(tmp_path, monkeypatch):
    from omlx.engine_core import get_mlx_executor

    calls, _ = _stub_loaders(monkeypatch)
    config = {"model_type": "llama"}
    source = _checkpoint(tmp_path / "source", config)
    walk = get_mlx_executor().submit(oq._module_classes, str(source), config)
    assert walk.result(timeout=60)["proj"][2] == (16, 64)
    assert calls[0][1].startswith("mlx-global")


def test_the_walk_restores_what_a_module_it_imports_holds(tmp_path, monkeypatch):
    imported = []
    real = oq.importlib.import_module

    def importing(name):
        imported.append(name)
        return real(name)

    monkeypatch.setattr(oq.importlib, "import_module", importing)
    _stub_loaders(monkeypatch)
    # Qwen4-Exp's runtime, as the pre-load patches set it for the model.
    language = SimpleNamespace(
        _MTP_RUNTIME="served", _PLE_RUNTIME_MODEL_PATH=None, _PLE_RUNTIME_MODE="r"
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.qwen4_exp.language", language)
    from omlx.utils import model_loading

    patches = model_loading.maybe_apply_pre_load_patches

    def setting(*args, **kwargs):
        patches(*args, **kwargs)
        language._MTP_RUNTIME, language._PLE_RUNTIME_MODE = "walked", "ssd"

    monkeypatch.setattr(model_loading, "maybe_apply_pre_load_patches", setting)
    config = {"model_type": "llama"}
    oq._module_classes(str(_checkpoint(tmp_path / "source", config)), config)
    assert (language._MTP_RUNTIME, language._PLE_RUNTIME_MODE) == ("served", "r")
    # oMLX's modules of that state are imported before the walk starts.
    assert {name for name, _ in oq._LOAD_GLOBALS if name.startswith("omlx.")} <= set(
        imported
    )


def test_the_walk_restores_qwen4_exp_s_runtime_it_registers(tmp_path, monkeypatch):
    from omlx.patches import mlx_vlm_qwen4_exp_compat
    from omlx.utils import model_loading

    _stub_loaders(monkeypatch)
    name = "mlx_vlm.models.qwen4_exp.language"
    monkeypatch.delitem(sys.modules, name, raising=False)
    language = SimpleNamespace(
        _MTP_RUNTIME="default", _PLE_RUNTIME_MODEL_PATH=None, _PLE_RUNTIME_MODE="r"
    )

    def registering():
        sys.modules[name] = language

    offloaded = []

    def setting(*args, **kwargs):
        # The dispatch registers Qwen4-Exp, then sets its runtime.
        registering()
        language._MTP_RUNTIME, language._PLE_RUNTIME_MODE = "walked", "ssd"
        offloaded.append(kwargs["model_settings"].qwen4_ple_ssd_offload)

    monkeypatch.setattr(
        mlx_vlm_qwen4_exp_compat, "apply_mlx_vlm_qwen4_exp_compat_patch", registering
    )
    monkeypatch.setattr(model_loading, "maybe_apply_pre_load_patches", setting)
    monkeypatch.setattr(oq, "_is_qwen4_exp_config", lambda config: True)
    try:
        config = {"model_type": "llama"}
        oq._module_classes(str(_checkpoint(tmp_path / "source", config)), config)
        assert (language._MTP_RUNTIME, language._PLE_RUNTIME_MODE) == ("default", "r")
        # Its tables are resident for the walk: no file is opened for them.
        assert offloaded == [False]
    finally:
        sys.modules.pop(name, None)


def test_the_walk_patches_the_mtp_runtime_of_its_model_type(tmp_path, monkeypatch):
    from omlx.patches import mlx_vlm_mtp

    calls, _ = _stub_loaders(monkeypatch)
    asked = []
    monkeypatch.setattr(
        mlx_vlm_mtp, "apply_mlx_vlm_mtp_runtime_patch", lambda t=None: asked.append(t)
    )
    monkeypatch.setattr(mlx_vlm_mtp, "apply_mlx_vlm_mtp_patch", lambda: True)
    config = {
        "model_type": "qwen3_vl",
        "vision_config": {"depth": 1},
        "mtp_num_hidden_layers": 1,
    }
    source = _checkpoint(tmp_path / "source", config, tensors=("mtp.fc.weight",))
    oq._module_classes(str(source), config)
    assert calls[0][0] == "mlx-vlm" and asked == ["qwen3_vl"]


def test_the_walk_is_of_the_mtp_the_run_keeps(source, toy, tmp_path, monkeypatch):
    layout = oq._written_layout
    kept = []

    def written(*args, **kwargs):
        kept.append(kwargs["preserve_mtp"])
        return layout(*args, **kwargs)

    monkeypatch.setattr(oq, "_written_layout", written)
    # A source the run keeps an MTP head of.
    monkeypatch.setattr(oq, "_source_has_nextn_tensors", lambda keys, config: True)
    _quantize(source, tmp_path / "oqe-toy", quant_format="toy", preserve_mtp=True)
    assert kept == [True]
    config = {"model_type": "qwen3_5_moe", "mtp_num_hidden_layers": 1}
    run = {"bits": 4, "group_size": 64}
    for preserve in (True, False):
        written = layout(
            config,
            tmp_path,
            text_only=False,
            preserve_mtp=preserve,
            dtype="bfloat16",
            quantization=run,
        )
        assert written["mtp_num_hidden_layers"] == (1 if preserve else 0)


def test_the_walk_loads_as_the_run_trusts_remote_code(
    source, toy, tmp_path, monkeypatch
):
    import mlx_lm.utils

    walk, loader = oq._module_classes, mlx_lm.utils.load_model
    asked, loaded = [], []

    def loading(*args, **kwargs):
        loaded.append(kwargs["trust_remote_code"])
        return loader(*args, **kwargs)

    def walking(*args, **kwargs):
        asked.append(kwargs["trust_remote_code"])
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(mlx_lm.utils, "load_model", loading)
            return walk(*args, **kwargs)

    monkeypatch.setattr(oq, "_module_classes", walking)
    _quantize(source, tmp_path / "oqe-toy", quant_format="toy", trust_remote_code=True)
    assert asked == [True] and loaded == [True]


def test_the_walk_is_of_the_config_the_run_writes(source, toy, tmp_path, monkeypatch):
    # A source whose config the run changes: text-only, without MTP layers
    # (the key Qwen3.5 does not read), in float16.
    changed = tmp_path / "changed"
    changed.mkdir()
    for item in source.iterdir():
        (changed / item.name).write_bytes(item.read_bytes())
    config = json.loads((source / "config.json").read_text())
    extra = {"torch_dtype": "bfloat16", "n_mtp_layers": 1}
    (changed / "config.json").write_text(
        json.dumps({**config, **extra, "image_token_id": 5})
    )
    walk = oq._module_classes
    given = []

    def walking(model_path, config, **kwargs):
        given.append(config)
        return walk(model_path, config, **kwargs)

    monkeypatch.setattr(oq, "_module_classes", walking)
    output = tmp_path / "oqe-toy"
    _, written, _ = _quantize(
        changed, output, quant_format="toy", dtype="float16", text_only=True
    )
    assert written["torch_dtype"] == "float16" and "image_token_id" not in written
    assert "n_mtp_layers" not in written
    # It is the config written, but what the run learns as it writes: its
    # layers' quantization entries, the formats' record, the tokenizer's end
    # of sequence.
    run = {k: v for k, v in written["quantization"].items() if not isinstance(v, dict)}
    learned = {"quantization": run, "quantization_config": run}
    unlearned = {k: v for k, v in written.items() if k not in (_RECORD, "eos_token_id")}
    assert given == [{**unlearned, **learned}]


def test_the_walk_of_minimax_nvfp4_experts_has_their_scales(tmp_path):
    config = {
        "model_type": "minimax_m3_vl",
        "text_config": {"n_shared_experts": 1, "pack_shared_expert": False},
        "quantization_config": {"quant_method": "mxfp8", "moe_quant_algo": "NVFP4"},
    }
    base = {"group_size": 64, "bits": 4, "mode": "affine"}
    written = oq._written_layout(
        config,
        tmp_path,
        text_only=False,
        preserve_mtp=False,
        dtype="bfloat16",
        quantization=base,
    )
    assert written["text_config"] == {
        "n_shared_experts": 1,
        "pack_shared_expert": False,
        "routed_expert_scales": True,
    }
    assert written["quantization"] == written["quantization_config"] == base
    assert config["text_config"] == {"n_shared_experts": 1, "pack_shared_expert": False}


def test_a_matrix_the_format_declines_outside_oq_s_groups_is_written_as_without(
    source, toy, tmp_path, monkeypatch, caplog
):
    # In groups of 128, the format takes the matrices oQ's groups divide and
    # declines those of K = 64, which are written in floats, as without it;
    # so are two of them that the format is not asked of: one that no module
    # holds, and one whose module's class no format's layer is built for.
    theirs = _quantize(source, tmp_path / "oqe", group_size=128)
    floats = sorted(
        name[: -len(".weight")]
        for name, tensor in theirs[0].items()
        if name.endswith("_proj.weight")
        and tensor.shape[-1] == 64
        and name.replace(".weight", ".scales") not in theirs[0]
    )
    unheld, unloaded, missing = floats[:3]
    walk = oq._module_classes

    def walking(*args, **kwargs):
        walked = walk(*args, **kwargs)
        del walked[unheld]
        cls, _, shape = walked[unloaded]
        return {**walked, unloaded: (cls, False, shape)}

    monkeypatch.setattr(oq, "_module_classes", walking)
    toy.answer = lambda tensors, w: None if w.shape[-1] % 128 else tensors
    caplog.set_level(logging.INFO, logger="omlx.oq")
    output = tmp_path / "oqe-toy"
    tensors, config, applied = _quantize(
        source, output, group_size=128, quant_format="toy"
    )
    entries = _in_format(config)
    assert entries and all(
        tensors[f"{name}.weight"].shape[-1] == 128 for name in entries
    )
    for name in (unheld, unloaded):
        assert tensors[f"{name}.weight"].shape[-1] == 64
    _assert_same_output(
        _beside(tensors, config, entries), _beside(*theirs[:2], entries)
    )
    # Neither the report nor the warning counts them.
    ours = json.loads((output / "oq_imatrix_report.json").read_text())
    plain = json.loads((tmp_path / "oqe" / "oq_imatrix_report.json").read_text())
    for key in ("applied", "missing", "mismatched", "zero_count_experts"):
        assert ours[key] == plain[key], key
    assert not any("stay oQ's affine" in message for message in caplog.messages)
    # Nor does a strict run stop at one of them that has no importance: a
    # format is only asked of it. (The calibration measures no lm_head here:
    # the cache is given an entry for each matrix it missed.)
    cache = tmp_path / "imatrix.npz"
    _quantize(source, tmp_path / "first", group_size=128, imatrix_cache_path=str(cache))
    with np.load(cache) as data:
        kept = {name: data[name] for name in data.files}
    assert f"{missing}.in_sum2" in kept
    kept = {k: v for k, v in kept.items() if not k.startswith(f"{missing}.")}
    wide = next(
        k[: -len(".in_sum2")]
        for k, v in kept.items()
        if k.endswith(".in_sum2") and v.shape == (64,)
    )
    first = json.loads((tmp_path / "first" / "oq_imatrix_report.json").read_text())
    for name in first["missing"]:
        for part in ("in_sum2", "counts"):
            kept[f"{name}.{part}"] = kept[f"{wide}.{part}"]
    np.savez_compressed(cache, **kept)
    monkeypatch.setattr(oq, "_collect_imatrix", lambda *a, **k: pytest.fail("ran"))
    _, config, _ = _quantize(
        source,
        tmp_path / "strict",
        group_size=128,
        quant_format="toy",
        imatrix_cache_path=str(cache),
        imatrix_strict=True,
    )
    assert missing not in _in_format(config)


def test_a_stack_outside_oq_s_groups_that_the_format_takes_is_in_the_report(
    source, toy, tmp_path, monkeypatch
):
    # An expert of a stack of K = 64 that no calibration token reached: in
    # groups of 128 the format alone writes the stack, and the report counts
    # it as it counts the format's other matrices.
    cache = tmp_path / "imatrix.npz"
    _quantize(source, tmp_path / "first", group_size=128, imatrix_cache_path=str(cache))
    gate = "language_model.model.layers.0.mlp.switch_mlp.gate_proj"
    with np.load(cache) as data:
        kept = {name: data[name] for name in data.files}
    counts = kept[f"{gate}.counts"].copy()
    counts[0] = 0
    np.savez_compressed(cache, **{**kept, f"{gate}.counts": counts})
    monkeypatch.setattr(oq, "_collect_imatrix", lambda *a, **k: pytest.fail("ran"))
    output = tmp_path / "oqe-toy"
    tensors, config, applied = _quantize(
        source,
        output,
        group_size=128,
        quant_format="toy",
        imatrix_cache_path=str(cache),
    )
    assert tensors[f"{gate}.weight"].shape == (4, 128, 64)
    assert gate in _in_format(config) and gate in applied
    report = json.loads((output / "oq_imatrix_report.json").read_text())
    assert report["zero_count_experts"] == 1


def _minimax_m3_vl(path, layers=2):
    """A tiny MiniMax M3 checkpoint in bfloat16 at ``path``, every weight
    real, its experts stored unpacked (mlx-vlm's names, a shared expert
    beside the routed stacks), and a config that leaves the layout to the
    model: one shared expert as wide as a routed one, so the model packs
    it with them when it loads."""
    from mlx_vlm.utils import get_model_and_args, update_module_configs

    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()
    text = {
        "hidden_size": 64,
        "intermediate_size": 32,
        "shared_intermediate_size": 32,
        "dense_intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "num_hidden_layers": layers,
        "num_local_experts": 2,
        "num_experts_per_tok": 1,
        "n_shared_experts": 1,
        "moe_layer_freq": [1] * layers,
        "vocab_size": 64,
    }
    config = {
        "model_type": "minimax_m3_vl",
        "text_config": text,
        "vision_config": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "projection_dim": 64,
        },
        "projector_hidden_size": 64,
        "vocab_size": 64,
    }
    unpacked = {**config, "text_config": {**text, "pack_shared_expert": False}}
    module, _ = get_model_and_args(config=unpacked)
    model_config = update_module_configs(
        module.ModelConfig.from_dict(json.loads(json.dumps(unpacked))),
        module,
        unpacked,
        ["text", "vision", "perceiver", "projector", "audio"],
    )
    mx.random.seed(11)
    model = module.Model(model_config)
    model.set_dtype(mx.bfloat16)
    path.mkdir()
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(path / "model.safetensors"), weights, {"format": "mlx"})
    (path / "config.json").write_text(json.dumps(config))
    _save_tokenizer(path, 64)
    return weights


def test_the_walk_keeps_the_modules_the_source_s_index_declares(tmp_path):
    # mlx-vlm drops a module beside the language model that no weight is
    # given for, unless the checkpoint's index declares it (an MTP head, a
    # vision tower): the walk declares what the source's index does.
    source = tmp_path / "minimax"
    weights = _minimax_m3_vl(source)
    config = json.loads((source / "config.json").read_text())

    def tower(walked):
        return [
            name
            for name, (_, loads, shape) in walked.items()
            if name.startswith("vision_tower.") and loads and shape is not None
        ]

    # One file of weights, no index: its tensors are declared from its header.
    assert not (source / "model.safetensors.index.json").exists()
    single = tower(oq._module_classes(str(source), config))
    assert single
    index = {"metadata": {}, "weight_map": dict.fromkeys(weights, "model.safetensors")}
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    assert tower(oq._module_classes(str(source), config)) == single


def test_an_unpacked_minimax_at_8_bits_writes_a_checkpoint_that_loads(
    toy, tmp_path, monkeypatch
):
    from mlx_vlm import utils as vlm_utils

    from omlx.engine.vlm import _force_minimax_m3_moe_sanitize_on_load

    source = tmp_path / "minimax"
    weights = _minimax_m3_vl(source)
    moe = "language_model.model.layers.0.block_sparse_moe"
    assert f"{moe}.shared_experts.down_proj.weight" in weights
    assert weights[f"{moe}.switch_mlp.down_proj.weight"].shape == (2, 64, 32)
    output = tmp_path / "oqe8-toy"
    tokens = (mx.arange(128).reshape(8, 16) % 63 + 1).astype(mx.int32)
    monkeypatch.setattr(oq, "_load_calibration_data", lambda *a, **k: tokens)
    # oMLX patches MiniMax M3: taken as verified, so that the run writes it.
    monkeypatch.setattr(
        oq, "_QUANT_FORMAT_VERIFIED", frozenset({"minimax_m3_vl"}), raising=False
    )
    oq.quantize_oq_streaming(
        str(source),
        str(output),
        8,
        enhanced=True,
        imatrix_num_samples=4,
        imatrix_seq_length=16,
        sensitivity_map_override={0: 0.1, 1: 0.2},
        quant_format="toy",
    )
    config = json.loads((output / "config.json").read_text())
    # The loader packs the shared expert with the routed stacks: none of the
    # experts' matrices is the format's.
    assert not any(".block_sparse_moe." in name for name in _in_format(config))
    with _force_minimax_m3_moe_sanitize_on_load(output):
        maybe_apply_pre_load_patches(str(output), for_vlm=True)
        model = vlm_utils.load_model(output)
    logits = model.language_model(mx.array([[1, 2, 3, 4]]))
    logits = getattr(logits, "logits", logits)
    assert mx.isfinite(logits).all()
