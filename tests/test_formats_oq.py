# SPDX-License-Identifier: Apache-2.0
"""oQ writes a checkpoint in a quant format (``quant_format``): what a format
is offered and what is checked of its answer, then a tiny checkpoint written
in a toy format and loaded back.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import shutil
import sys
import textwrap
import threading
import weakref
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
from omlx.formats import testing
from omlx.formats.linear import FormatLinear, FormatSwitchLinear
from omlx.patches.format_load import loads_in_format
from omlx.utils.model_loading import maybe_apply_pre_load_patches
from tests.test_formats_kernels import _BLOCKS, _PAGE
from tests.test_formats_linear import _ToyFormat

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

_RECORD = "omlx_quant_formats"  # in a checkpoint's config: what reads its layers
_VERIFICATION = "omlx_verification"  # in its config: its run's verification
# docs/quant-formats.md quotes what oQ says of a format, one named myformat.
_QUOTED = " ".join(_PAGE.split())


def _quoted(message, name="toy"):
    return message.replace(name, "myformat") in _QUOTED


# The families verified with quant formats, as this oMLX ships them.
_VERIFIED = oq._QUANT_FORMAT_VERIFIED


@pytest.fixture(autouse=True)
def verified():
    """The toy checkpoint's family taken as verified: oMLX patches Qwen3.5
    MoE on an mlx-vlm load, and oQ writes a family it patches in a format
    only once the family is verified."""
    with testing.verified("llama", "qwen3_5_moe", "qwen3_5_moe_text"):
        yield


class _Log(SimpleNamespace):
    """One log for a format and every copy of it a load accepts with."""

    def __deepcopy__(self, memo):
        return self


class _Codec(_ToyFormat):
    """The toy format with its codec: a byte for each weight and a scale for
    each row, which keep a weight to half a scale. ``answer`` stands in for
    a codec that declines, or has a defect."""

    def __init__(self):
        super().__init__()
        self.offered, self.encoded, self.lays_out = [], [], True
        self.answer = lambda tensors, w: tensors
        # What the loads handed it, and the copies they accepted with.
        self.log = _Log(accepted=[])

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


def _matrix_bytes(n, k, dtype=mx.float32):
    """What one (n, k) matrix holds while the toy encodes it: its input in
    ``dtype``, a byte for each weight and a float32 scale for each row."""
    return n * k * dtype.size + n * k + n * 4


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
    # Two of its matrices fit a launch: their float32 input and their tensors.
    monkeypatch.setattr(oq, "_QUANTIZE_CHUNK_BYTES", 2 * _matrix_bytes(n, k))
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
    monkeypatch.setattr(oq, "_QUANTIZE_CHUNK_BYTES", _matrix_bytes(8, 64) - 1)
    assert offer(8, 64) is None and offer(3, 8, 64) is None
    assert toy.encoded == []
    # A matrix of a launch's size is, and a stack of them one at a time.
    monkeypatch.setattr(oq, "_QUANTIZE_CHUNK_BYTES", _matrix_bytes(8, 64))
    assert offer(8, 64).keys() == {"weight", "scales"}
    assert offer(3, 8, 64)["weight"].shape == (3, 8, 64)
    assert [given.shape[0] for given, *_ in toy.encoded] == [8, 1, 1, 1]
    # MLX's own limit on a launch holds whatever the buffer's.
    monkeypatch.setattr(oq, "_QUANTIZE_CHUNK_BYTES", 1 << 40)
    huge = SimpleNamespace(shape=((1 << 15) + 1, 1 << 15))
    assert oq._encode_in_format(toy, "a.weight", huge, importance, 4, 64) is None


@pytest.mark.parametrize(
    "dtype,runs", [(mx.bfloat16, [4, 4, 2]), (mx.float32, [2, 2, 2, 2, 2])]
)
def test_a_run_counts_its_input_in_its_own_dtype(toy, monkeypatch, dtype, runs):
    """A run holds four matrices of a bfloat16 stack, two of a float32
    one: its input in the stack's dtype and its tensors in the layout's."""
    monkeypatch.setattr(
        oq, "_QUANTIZE_CHUNK_BYTES", 4 * _matrix_bytes(8, 64, mx.bfloat16)
    )
    w = _weights(10, 8, 64).astype(dtype)
    kept = oq._encode_in_format(toy, "experts.weight", w, mx.ones((64,)), 4, 64)
    assert [given.shape[0] for given, *_ in toy.encoded] == runs
    assert _same(kept, _Codec().encode(w, mx.ones((64,)), 4, 64))


def test_a_stack_whose_tensor_would_not_fit_a_metal_buffer_is_not_offered(
    toy, monkeypatch
):
    """A stack's tensor is one array: its joined codes, 3 x 8 x 64 bytes,
    must fit a Metal buffer."""
    w, importance = _weights(3, 8, 64), mx.ones((64,))
    monkeypatch.setattr(oq, "_METAL_MAX_BUFFER", 3 * 8 * 64 - 1)
    assert oq._encode_in_format(toy, "experts.weight", w, importance, 4, 64) is None
    assert toy.encoded == []
    monkeypatch.setattr(oq, "_METAL_MAX_BUFFER", 3 * 8 * 64)
    kept = oq._encode_in_format(toy, "experts.weight", w, importance, 4, 64)
    assert kept["weight"].shape == (3, 8, 64)


def test_a_stack_s_runs_are_let_go_one_by_one(toy, monkeypatch):
    """Each run's tensors are evaluated, the GPU waited for and MLX's cache
    given back before the next run is read: a stack read from its
    checkpoint a run at a time holds one run, not the stack."""
    monkeypatch.setattr(oq, "_QUANTIZE_CHUNK_BYTES", 2 * _matrix_bytes(8, 64))
    answers, events = [], []

    def answer(tensors, w):
        answers.append(tensors)
        events.append("encode")
        return tensors

    toy.answer = answer
    evaluate = mx.eval

    def evaluated(*arrays):
        # Every tensor of the run's answer.
        if answers and all(any(a is t for a in arrays) for t in answers[-1].values()):
            events.append("eval")
        return evaluate(*arrays)

    chunks = oq._row_chunks_with_bounds

    def read(t, max_elems):
        for chunk in chunks(t, max_elems):
            events.append("read")
            yield chunk

    def recorded(name, call):
        def record():
            events.append(name)
            return call()

        return record

    monkeypatch.setattr(mx, "eval", evaluated)
    monkeypatch.setattr(oq, "_row_chunks_with_bounds", read)
    for name in ("synchronize", "clear_cache"):
        monkeypatch.setattr(mx, name, recorded(name, getattr(mx, name)))
    w = _weights(5, 8, 64)
    oq._encode_in_format(toy, "experts.weight", w, mx.ones((64,)), 4, 64)
    assert events == ["read", "encode", "eval", "synchronize", "clear_cache"] * 3


def test_a_run_holds_at_most_a_launch_s_weights(toy, monkeypatch):
    """Whatever the chunk, a run is at most _LAUNCH_WEIGHTS weights (MLX
    indexes with 32-bit integers), and a matrix of more is not offered."""
    monkeypatch.setattr(oq, "_QUANTIZE_CHUNK_BYTES", 1 << 40)
    monkeypatch.setattr(oq, "_LAUNCH_WEIGHTS", 2 * 8 * 64)
    importance = mx.ones((64,))
    w = _weights(5, 8, 64)
    kept = oq._encode_in_format(toy, "experts.weight", w, importance, 4, 64)
    assert [given.shape[0] for given, *_ in toy.encoded] == [2, 2, 1]
    assert _same(kept, _Codec().encode(w, importance, 4, 64))
    toy.encoded.clear()

    def offer(*shape):
        return oq._encode_in_format(
            toy, "a.weight", _weights(*shape), importance, 4, 64
        )

    # A matrix of a launch's weights is offered, one of more is not.
    assert offer(16, 64) and offer(17, 64) is None
    assert [given.shape for given, *_ in toy.encoded] == [(16, 64)]


def test_runs_are_counted_in_matrices_of_any_shape(toy, monkeypatch):
    monkeypatch.setattr(oq, "_QUANTIZE_CHUNK_BYTES", 2 * _matrix_bytes(16, 32))
    w = _weights(5, 16, 32)
    oq._encode_in_format(toy, "experts.weight", w, mx.ones((32,)), 4, 32)
    assert [given.shape[0] for given, *_ in toy.encoded] == [2, 2, 1]


@pytest.mark.parametrize("shape", [(0, 8, 64), (0, 64), (3, 0, 64), (8, 0)])
def test_an_empty_matrix_or_stack_is_not_offered(toy, shape):
    w = mx.zeros(shape)
    assert oq._encode_in_format(toy, "a.weight", w, mx.ones((64,)), 4, 64) is None
    # Not asked of the format: its layout may divide by an axis.
    assert toy.offered == [] and toy.encoded == []


def test_a_name_s_runs_are_let_go_once_it_is_joined(toy, monkeypatch):
    """A stack's runs are joined name by name, each name's run tensors held
    by its own join alone: evaluating it lets them go."""
    monkeypatch.setattr(oq, "_QUANTIZE_CHUNK_BYTES", _matrix_bytes(64, 512))
    w = _weights(4, 64, 512)
    kept = oq._encode_in_format(toy, "experts.weight", w, mx.ones((512,)), 4, 64)
    mx.synchronize()
    before = mx.get_active_memory()
    mx.eval(kept["weight"])
    mx.synchronize()
    # The codes, a byte a weight, joined and their runs let go: as much held.
    assert mx.get_active_memory() - before < 4 * 64 * 512 // 2


def test_a_stack_in_one_run_is_kept_as_it_was_encoded(toy):
    """A stack that one run holds is not joined: oQ keeps the tensors the
    format gave."""
    answers = []
    toy.answer = lambda tensors, w: answers.append(tensors) or tensors
    w = _weights(3, 8, 64)
    kept = oq._encode_in_format(toy, "experts.weight", w, mx.ones((64,)), 4, 64)
    assert len(answers) == 1
    assert all(kept[name] is tensor for name, tensor in answers[0].items())


def test_rows_are_cut_from_a_tensor_of_no_rows_or_of_empty_rows():
    """oQ's row chunks (its affine writer's and a format's runs): none of a
    tensor without rows; of a tensor whose rows are empty, each row counts
    one element."""
    assert list(oq._row_chunks_with_bounds(mx.zeros((0, 4)), 8)) == []
    chunks = oq._row_chunks_with_bounds(mx.zeros((3, 0)), 3)
    assert [(r0, r1, chunk.shape) for r0, r1, chunk in chunks] == [(0, 3, (3, 0))]
    chunks = oq._row_chunks_with_bounds(mx.zeros((3, 0)), 2)
    assert [(r0, r1) for r0, r1, _ in chunks] == [(0, 2), (2, 3)]


def test_a_run_in_a_format_writes_beside_its_output_until_done(
    source, toy, tmp_path, monkeypatch
):
    """A run with a quant format writes into <output>.unverified, which no
    scan serves, and renames it to <output> once written."""
    output = tmp_path / "oqe-toy"
    unverified = tmp_path / "oqe-toy.unverified"
    copy, seen = oq._copy_model_sidecars, []

    def copied(src, out, **kwargs):
        # The run's last writes: where it writes them.
        seen.append((out, output.exists()))
        return copy(src, out, **kwargs)

    monkeypatch.setattr(oq, "_copy_model_sidecars", copied)
    _quantize(source, output, quant_format="toy")
    assert seen == [(unverified, False)]
    assert output.is_dir() and not unverified.exists()
    # Without a format, the run writes straight into its output.
    seen.clear()
    _quantize(source, tmp_path / "oqe")
    assert seen == [(tmp_path / "oqe", True)]


def test_a_run_in_a_format_records_its_verification(source, toy, tmp_path, monkeypatch):
    """Not verified while it writes, so a crash leaves what a load refuses;
    verified, by the version of the checks it passed, once done."""
    output = tmp_path / "oqe-toy"
    copy, seen = oq._copy_model_sidecars, []

    def copied(src, out, **kwargs):
        seen.append(json.loads((out / "config.json").read_text())[_VERIFICATION])
        return copy(src, out, **kwargs)

    monkeypatch.setattr(oq, "_copy_model_sidecars", copied)
    _, config, _ = _quantize(source, output, quant_format="toy")
    assert seen == [
        {
            "verified": False,
            "mode": "toy",
            "verify_version": oq.VERIFY_VERSION,
            "reason": "its quantization did not complete",
        }
    ]
    assert config[_VERIFICATION] == {
        "verified": True,
        "mode": "toy",
        "verify_version": oq.VERIFY_VERSION,
    }


def test_a_run_in_a_format_keeps_the_source_s_reference(source, toy, tmp_path):
    """Made once before it writes, beside its calibration's cache, and
    copied into the output; another run of the same source and tokenizer
    reuses it."""
    from omlx import oq_verify

    made = []
    reference_of = oq_verify.reference_of

    def making(model, prompt, layers):
        made.append(prompt.tolist())
        return reference_of(model, prompt, layers)

    cached = tmp_path / ".oqe_imatrix" / f"{source.name}.reference.npz"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(oq_verify, "reference_of", making)
        # A run without a format makes none.
        _quantize(source, tmp_path / "c")
        assert made == [] and not cached.exists()
        _quantize(source, tmp_path / "a", quant_format="toy")
        _quantize(source, tmp_path / "b", quant_format="toy")
        tokens = (mx.arange(16) % 63 + 1).tolist()
        assert made == [tokens]
        # The same source with another tokenizer: made again.
        other = tmp_path / "other" / source.name
        shutil.copytree(source, other)
        told = json.loads((other / "tokenizer_config.json").read_text())
        (other / "tokenizer_config.json").write_text(
            json.dumps({**told, "model_max_length": 7})
        )
        _quantize(other, tmp_path / "d", quant_format="toy")
        assert made == [tokens, tokens]
    kept = [np.load(cached)] + [
        np.load(tmp_path / name / "verify" / "reference.npz") for name in ("a", "b")
    ]
    for reference in kept:
        assert reference["prompt"].tolist() == tokens
        assert reference["ids"].shape == (16, 16)
        np.testing.assert_array_equal(reference["logprobs"], kept[0]["logprobs"])
    assert not (tmp_path / "c" / "verify").exists()


def test_a_run_keeps_no_record_of_its_source_s_verification(source, tmp_path):
    """A source written by a run in a format tells of that run's checks,
    not of this one's."""
    copy = tmp_path / "written"
    shutil.copytree(source, copy)
    config = json.loads((copy / "config.json").read_text())
    config[_VERIFICATION] = {"verified": True, "mode": "toy", "verify_version": 0}
    (copy / "config.json").write_text(json.dumps(config))
    _, config, _ = _quantize(copy, tmp_path / "oqe")
    assert _VERIFICATION not in config


def test_a_run_in_a_format_refuses_a_folder_it_would_write(source, toy, tmp_path):
    output = tmp_path / "oqe-toy"
    unverified = tmp_path / "oqe-toy.unverified"
    unverified.mkdir()
    with pytest.raises(ValueError) as refusal:
        oq.quantize_oq_streaming(
            str(source), str(output), 4, enhanced=True, quant_format="toy"
        )
    assert str(refusal.value) == f"Output directory already exists: {unverified}"
    assert not output.exists() and list(unverified.iterdir()) == []
    assert toy.offered == []


def test_a_stack_s_join_is_evaluated_once_the_writer_lets_its_source_go(
    source, toy, tmp_path, monkeypatch
):
    """The writer keeps a layer's tensors until its shard is saved: it
    evaluates a stack's joined tensors, one name at a time, once it has let
    the source stack go, so that a join does not overlap it, and waits for
    the GPU after each, so that its runs go before the next; the format's
    encoding returns them unevaluated."""
    # Runs of one matrix: every stack of experts is joined.
    monkeypatch.setattr(oq, "_QUANTIZE_CHUNK_BYTES", 1 << 16)
    encode, evaluate, joins, evaluated = oq._encode_in_format, mx.eval, {}, []
    synchronize, events = mx.synchronize, []
    # What is evaluated while the format encodes; held for that call alone,
    # so that nothing here keeps the source alive.
    window = []

    def watched(*arrays):
        if window:
            window[-1].append(arrays)
        events.append("eval")
        for array in arrays:
            joined = joins.get(id(array))
            if joined is not None and joined[0] is array:
                evaluated.append((len(arrays), joined[1]() is None))
                events[-1] = "join"
        return evaluate(*arrays)

    def synchronized(*args, **kwargs):
        events.append("synchronize")
        return synchronize(*args, **kwargs)

    def encoding(fmt, tensor_name, w, *args):
        window.append([])
        runs = len(toy.encoded)
        tensors = encode(fmt, tensor_name, w, *args)
        during = window.pop()
        if tensors is not None and len(toy.encoded) - runs > 1:
            for tensor in tensors.values():
                # Not evaluated while the source is held.
                assert not any(tensor is a for arrays in during for a in arrays)
                joins[id(tensor)] = (tensor, weakref.ref(w))
        return tensors

    monkeypatch.setattr(oq, "_encode_in_format", encoding)
    monkeypatch.setattr(mx, "eval", watched)
    monkeypatch.setattr(mx, "synchronize", synchronized)
    _quantize(source, tmp_path / "oqe-toy", quant_format="toy")
    # Each alone, each once the source stack is gone, each once.
    assert evaluated and set(evaluated) == {(1, True)}
    assert len(evaluated) == len(joins)
    # A join's runs are let go before anything else is evaluated.
    after = [events[i + 1] for i, event in enumerate(events) if event == "join"]
    assert after == ["synchronize"] * len(joins)


def test_a_matrix_the_format_lays_out_none_of_is_not_offered(toy):
    toy.lays_out = False
    w = _weights(8, 64)
    assert oq._encode_in_format(toy, "a.weight", w, mx.ones((64,)), 4, 64) is None
    assert toy.offered == [(64, 8, 4, 64)] and toy.encoded == []


def test_a_format_declines_a_matrix_or_a_whole_stack(toy, monkeypatch):
    importance = mx.ones((64,))
    monkeypatch.setattr(oq, "_QUANTIZE_CHUNK_BYTES", 2 * _matrix_bytes(8, 64))
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


# Layouts that the toy's encode is made to agree with, and that no layer
# holds, with what oQ says of each for an (8, 64) matrix.
_UNHELD = {
    "a tensor more": (
        lambda n, k: {
            "weight": ((n, k), mx.uint8),
            "scales": ((n,), mx.float32),
            "biases": ((n,), mx.float32),
        },
        "quant format 'toy' lays out a (8, 64) matrix as biases, scales, weight: "
        "a layer holds weight, scales",
    ),
    "one axis": (
        lambda n, k: {"weight": ((n * k,), mx.uint8), "scales": ((n,), mx.float32)},
        "quant format 'toy' lays out the weight of a (8, 64) matrix as (512,): a "
        "weight has two axes, N and the format's own",
    ),
    "packed along N": (
        lambda n, k: {
            "weight": ((n // 2, 2 * k), mx.uint8),
            "scales": ((n,), mx.float32),
        },
        "quant format 'toy' lays out the weight of a (8, 64) matrix as (4, 128): "
        "its first axis is N (8); a format packs along K, never along N",
    ),
}


def _lays_out(toy, monkeypatch, layout):
    """The toy lays matrices out as ``layout``, and encodes them so."""
    monkeypatch.setattr(toy, "layout", lambda k, n, bits, group_size: layout(n, k))

    def agreeing(tensors, w):
        *lead, n, k = w.shape
        return {
            name: mx.zeros((*lead, *shape), dtype)
            for name, (shape, dtype) in layout(n, k).items()
        }

    toy.answer = agreeing


@pytest.mark.parametrize("shape", [(8, 64), (3, 8, 64)], ids=["matrix", "stack"])
@pytest.mark.parametrize("case", list(_UNHELD))
def test_a_layout_no_layer_holds_is_refused_before_a_matrix_is_encoded(
    toy, monkeypatch, case, shape
):
    layout, words = _UNHELD[case]
    _lays_out(toy, monkeypatch, layout)
    with pytest.raises(ValueError, match=f"^{re.escape(words)}$"):
        oq._encode_in_format(toy, "a.weight", _weights(*shape), mx.ones((64,)), 4, 64)
    assert toy.encoded == []


def test_the_contract_page_quotes_the_refusal_of_a_layout(toy, monkeypatch):
    monkeypatch.setattr(toy, "mode", "myformat")
    _lays_out(toy, monkeypatch, _UNHELD["packed along N"][0])
    with pytest.raises(ValueError) as refusal:
        oq._encode_in_format(toy, "a.weight", _weights(16, 64), mx.ones((64,)), 8, 1)
    assert str(refusal.value) in _QUOTED


def test_the_contract_page_says_who_refuses_a_layout():
    assert (
        "A layout that breaks these rules for a matrix is refused, by oQ before "
        "it encodes the matrix and by a layer when it is built" in _QUOTED
    )


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


# oQe's settings of the tiny runs: four samples of 16 tokens, a sensitivity
# map given.
_SETTINGS = {
    "imatrix_num_samples": 4,
    "imatrix_seq_length": 16,
    "sensitivity_map_override": {0: 0.1, 1: 0.2},
}


def _calibration(patch, sampled=None):
    """oQe's calibration tokens for the tiny runs; ``sampled`` counts the
    times they are read."""
    tokens = (mx.arange(128).reshape(8, 16) % 63 + 1).astype(mx.int32)

    def load(*args, **kwargs):
        if sampled is not None:
            sampled.append(1)
        return tokens

    patch.setattr(oq, "_load_calibration_data", load)


def _quantize(source, output, sampled=None, **kwargs):
    """oQ4e of ``source``: the output's tensors, its config, and the layers
    that oQe had an importance for."""
    with pytest.MonkeyPatch.context() as patch:
        _calibration(patch, sampled)
        oq.quantize_oq_streaming(
            str(source), str(output), 4, enhanced=True, **_SETTINGS, **kwargs
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
    without their entries, the formats' record and the verification's."""
    tensors = {
        name: tensor
        for name, tensor in tensors.items()
        if name.rpartition(".")[0] not in layers
    }
    config = {
        name: value
        for name, value in config.items()
        if name not in (_RECORD, _VERIFICATION)
    }
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
    # As without it, but for the record of its verification: a run in a
    # format is checked whatever the format took.
    assert config[_VERIFICATION]["verified"] is True
    _assert_same_output(_beside(tensors, config, ()), plain[:2])


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
_NOT_VERIFIED = "it is no family verified with quant formats end to end"
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
        # Families the static reader of 57e0cbf missed: not verified, they
        # are refused.
        ("minicpmv4_7", {"model_type": "minicpmv4_7"}, _NOT_VERIFIED),
        ("olmoe", {"model_type": "olmoe"}, _NOT_VERIFIED),
        ("qwen3_omni_moe", {"model_type": "qwen3_omni_moe"}, _NOT_VERIFIED),
        ("xlm_roberta", {"model_type": "xlm-roberta"}, _NOT_VERIFIED),
        # One nobody has heard of: the gate fails closed.
        ("brand_new", {"model_type": "brand-new"}, _NOT_VERIFIED),
        # A verified top level (llama, in these tests) with a text model that
        # is not.
        (
            "gemma4_text",
            {"model_type": "llama", "text_config": {"model_type": "gemma4_text"}},
            _NOT_VERIFIED,
        ),
        (
            # A decision model by its model type, whatever its sidecars:
            # served as one, its loader refuses it (format_load.refuse_family).
            "qwen3_5",
            {"model_type": "qwen3_5"},
            _NOT_VERIFIED,
        ),
        (
            "inkling2",
            {"model_type": "inkling2"},
            "its DSpark draft folds the output head into its own projection by "
            "hand, as MLX's quantized layer or as floats",
        ),
        # No model type: the gate fails closed.
        ("untyped", {}, "its config names no model type"),
        ("untyped", {"model_type": None}, "its config names no model type"),
        ("untyped", {"model_type": " "}, "its config names no model type"),
        # A decision model's backbone under another spelling.
        (
            "lfm3_vl",
            {"model_type": "LFM3-VL", "architectures": ["D1Model"]},
            _NOT_VERIFIED,
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
        "text model not verified",
        "decision model",
        "inkling2",
        "no model type",
        "a null model type",
        "a blank model type",
        "decision model spelled otherwise",
    ],
)
def test_a_family_not_verified_is_refused(toy, tmp_path, family, config, why):
    source = tmp_path / "source"
    source.mkdir()
    if family == "qwen3_5":
        # Clef's joint head beside it.
        for name in ("joint_head.safetensors", "joint_head_config.json"):
            (source / name).write_text("{}")
    (source / "config.json").write_text(json.dumps(config))
    # Weights that no load reads: refused before any is.
    (source / "model.safetensors").write_bytes(b"not a checkpoint")
    output = tmp_path / "refused"
    with pytest.raises(formats.FormatRefusedError) as refusal:
        oq.quantize_oq_streaming(
            str(source), str(output), 4, enhanced=True, quant_format="toy"
        )
    fix = "name its model type in its config" if family == "untyped" else None
    assert str(refusal.value) == (
        f"quant format 'toy': oQ writes no {family} checkpoint in it: {why}; "
        f"write it without a quant format, or {fix or 'have the family verified'}"
    )
    assert refusal.value.mode == "toy" and refusal.value.path is None
    # The page quotes it, of the same family or one of its kind.
    said = str(refusal.value)
    for spelled, quoted in _PAGE_FAMILY.items():
        said = said.replace(spelled, quoted)
    quoted_spelling = config.get("model_type") in (family, "glm5", "qwen3_5")
    if family in _QUOTED_FAMILIES and quoted_spelling:
        assert _quoted(said)
    assert not output.exists() and toy.offered == []


def test_no_family_is_verified_but_by_a_test(tmp_path, monkeypatch):
    # Outside every block (this file's tests take llama as verified).
    monkeypatch.setattr(oq, "_TEST_VERIFIED", frozenset())
    for model_type in ("llama", " Llama ", "qwen2"):
        refused = oq._unverified_family({"model_type": model_type})
        assert refused == (oq._family(model_type), _NOT_VERIFIED)
    # Spaces and a blank text model type: the family the audits read.
    blank = {"model_type": " raon1_mini ", "text_config": {"model_type": " "}}
    assert oq._unverified_family(blank)[0] == "raon1_mini"
    with testing.verified("raon1_mini"):
        assert oq._unverified_family(blank) is None


@pytest.mark.parametrize("error", ["refused", "unwrapped"])
def test_oq_s_own_loads_let_a_refusal_through(tmp_path, monkeypatch, error):
    import omlx.utils.model_loading as model_loading

    raised = (
        formats.FormatRefusedError("refused", mode="toy")
        if error == "refused"
        else formats.LoadHookError("mlx_lm.utils.load_model is no function")
    )

    def load(*args, **kwargs):
        raise raised

    monkeypatch.setattr(model_loading, "lm_load_compat", load)
    config = {"model_type": "llama"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    # The calibration's load and both sensitivity measures': not a failure
    # to measure, which they go on without.
    for measure in (
        lambda: oq._calibration_model(str(tmp_path), config),
        lambda: oq._measure_sensitivity(str(tmp_path), config, 4),
        lambda: oq._measure_sensitivity_from_quantized_model(str(tmp_path), config, 4),
    ):
        with pytest.raises(type(raised)) as caught:
            measure()
        assert caught.value is raised


@pytest.mark.parametrize("mode", ["affine", "mxfp4"])
def test_one_of_mlx_s_modes_is_written_without_a_quant_format(tmp_path, mode):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with pytest.raises(ValueError, match="is one of MLX's own modes") as refusal:
        oq.quantize_oq_streaming(
            str(source), str(tmp_path / "out"), 4, enhanced=True, quant_format=mode
        )
    assert _quoted(str(refusal.value), name="no such name") is (mode == "mxfp4")


def test_the_families_oq_writes_in_a_format(tmp_path):
    # Verified ones only, and none is yet: Qwen3.8 Flash-Next enters with the
    # run that verifies it. Families mlx-lm's own code serves come back after
    # the proof of concept (DEFERRED.md row 22).
    assert not _VERIFIED and not hasattr(oq, "_QUANT_FORMAT_STOCK")
    def unverified(config):
        return oq._unverified_family(config)

    # (This file's tests take llama as verified, for their tiny checkpoints.)
    for model_type in ("Mistral", "qwen2", "PHI3", "starcoder2", "olmo2"):
        assert unverified({"model_type": model_type})[1] == _NOT_VERIFIED
    # A verified family, top level or text model, under any spelling.
    nested = {"model_type": "mistral", "text_config": {"model_type": "qwen4_exp"}}
    with testing.verified("Qwen4-Exp", "mistral"):
        assert unverified({"model_type": "qwen4_exp"}) is None
        assert unverified(nested) is None
    # Not after the block; nor a verified text model under another top level.
    assert unverified(nested)[0] == "mistral"
    with testing.verified("qwen4_exp"):
        assert unverified(nested)[0] == "mistral"
    assert unverified({"model_type": "qwen3"}) == ("qwen3", _NOT_VERIFIED)


def test_a_test_takes_families_as_verified_for_its_block_or_itself(tmp_path):
    import asyncio

    before = oq._TEST_VERIFIED
    config = {"model_type": "qwen4_exp"}

    @testing.verified("QWEN4-EXP")
    def written():
        return oq._unverified_family(config)

    @testing.verified("qwen4_exp")
    async def written_async():
        await asyncio.sleep(0)
        return oq._unverified_family(config)

    assert written() is None and asyncio.run(written_async()) is None
    assert before == oq._TEST_VERIFIED
    with pytest.raises(RuntimeError), testing.verified("qwen4_exp", "d1"):
        assert before | {"qwen4_exp", "d1"} == oq._TEST_VERIFIED
        raise RuntimeError
    assert before == oq._TEST_VERIFIED
    assert oq._unverified_family(config)[0] == "qwen4_exp"
    # A family the audits found misread is written too, when a test says so.
    with testing.verified("raon1_mini"):
        assert oq._unverified_family({"model_type": "raon1_mini"}) is None
    # Not bare, and not over a fixture's generator.
    with pytest.raises(TypeError, match="not a bare @verified"):
        testing.verified(written)
    with pytest.raises(TypeError, match="a fixture opens a block"):
        testing.verified("d1")(lambda: (yield))


def test_blocks_of_verified_close_in_any_order_in_any_thread(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    opened, closed = threading.Event(), threading.Event()

    def written(model_type):
        return oq._unverified_family({"model_type": model_type}) is None

    def other():
        with testing.verified("minimax_m3_vl"):
            opened.set()
            assert closed.wait(60)
            # The other thread's block closed first: this one's holds.
            return written("minimax_m3_vl"), written("qwen4_exp")

    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            with testing.verified("qwen4_exp"):
                running = pool.submit(other)
                assert opened.wait(60)
                assert written("qwen4_exp") and written("minimax_m3_vl")
        finally:
            closed.set()
        assert running.result(timeout=60) == (True, False)
    assert not written("qwen4_exp") and not written("minimax_m3_vl")


def test_verified_decorates_test_functions_only(monkeypatch, tmp_path):
    with pytest.raises(TypeError, match="decorates a test function"):

        @testing.verified("qwen4_exp")
        class TestToyFormat:
            pass

    with pytest.raises(TypeError, match="decorates a test function"):
        testing.verified("qwen4_exp")(pytest.fixture(lambda: None))
    with pytest.raises(TypeError, match="not a bare"):
        testing.verified(" ")
    # What oMLX ships as verified stays so, inside a block and out.
    monkeypatch.setattr(oq, "_QUANT_FORMAT_VERIFIED", frozenset({"qwen4_exp"}))
    config = {"model_type": "qwen4_exp"}
    assert oq._unverified_family(config) is None
    with testing.verified("qwen4_exp"):
        assert oq._unverified_family(config) is None
    assert oq._unverified_family(config) is None


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


@testing.verified("llama")
def test_a_verified_model_is_written_in_a_format(llama_source, toy, tmp_path):
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
    # The run hands it back to the format as the load does, before it
    # reads a weight; the load, once.
    assert toy.accepted == [record]
    maybe_apply_pre_load_patches(str(output))
    model, _ = load_model(output, strict=True)
    assert toy.accepted == [record, record]
    layers = [m for _, m in model.named_modules() if isinstance(m, FormatLinear)]
    # Read with the load's copy of the format, one for all of them.
    (read,) = {id(layer.fmt): layer.fmt for layer in layers}.values()
    assert read is not toy and type(read) is _Codec and read.log is toy.log


def test_the_record_names_the_contract_the_format_states(
    source, toy, tmp_path, monkeypatch
):
    """A checkpoint keeps the revision of the contract its format states,
    not the newest this oMLX knows."""
    # A revision between the oldest and the newest: neither is written for it.
    monkeypatch.setattr(formats, "CONTRACT", 3)
    monkeypatch.setattr(toy, "contract", 2)
    _, config, _ = _quantize(source, tmp_path / "oqe-toy", quant_format="toy")
    assert config[_RECORD]["toy"]["contract"] == 2


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
        {"scale": float("inf")},
        {"name": "\ud800"},
        _CYCLE,
    ],
    ids=[
        "no dict",
        "a tuple",
        "a key not a string",
        "an array",
        "a nan",
        "an infinity",
        "no utf-8",
        "a cycle",
    ],
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


def test_a_checkpoint_keeps_the_record_as_it_was_checked(
    source, toy, tmp_path, monkeypatch
):
    """A record the format changes after oQ asked for it (here, as it
    encodes) is stored as it was when oQ checked it."""
    record = {"layout": 2, "encoded": 0}
    monkeypatch.setattr(toy, "record", lambda: record)

    def answer(tensors, w):
        record["encoded"] += 1
        return tensors

    toy.answer = answer
    _, config, _ = _quantize(source, tmp_path / "oqe-toy", quant_format="toy")
    assert record["encoded"] > 0
    assert config[_RECORD]["toy"]["record"] == {"layout": 2, "encoded": 0}


def test_oq_writes_with_a_format_object_of_the_registered_class(
    source, toy, tmp_path, monkeypatch
):
    """A plugin's command hands oQ its configured object (a fitted table):
    oQ encodes with that object and stores its record, under its mode and
    with its plugin."""
    monkeypatch.setitem(formats._ORIGINS, "toy", "toy-plugin 0.1")
    fitted = _Codec()
    fitted.record = lambda: {"levels": [-2, 1]}
    tensors, config, applied = _quantize(
        source, tmp_path / "oqe-fitted", quant_format=fitted
    )
    assert len(fitted.encoded) == len(applied) and toy.encoded == []
    assert config[_RECORD] == {
        "toy": {
            "plugin": "toy-plugin 0.1",
            "contract": 1,
            "record": {"levels": [-2, 1]},
        }
    }
    assert sorted(_in_format(config)) == applied


class _Fitted(_Codec):
    """A subclass of the registered class: another accept() at load."""


def test_a_format_object_of_another_class_is_refused(source, toy, tmp_path):
    output = tmp_path / "refused"
    for given, name in ((_ToyFormat(), "_ToyFormat"), (_Fitted(), "_Fitted")):
        with pytest.raises(ValueError) as refusal:
            oq.quantize_oq_streaming(
                str(source), str(output), 4, enhanced=True, quant_format=given
            )
        assert str(refusal.value) == (
            f"quant format 'toy' is registered as _Codec, not {name}: pass an "
            "object of the registered class"
        )
    unregistered = _Codec()
    unregistered.mode = "lloyd"
    with pytest.raises(ValueError) as refusal:
        oq.quantize_oq_streaming(
            str(source), str(output), 4, enhanced=True, quant_format=unregistered
        )
    assert str(refusal.value) == (
        "no quant format is registered for mode 'lloyd': install its plugin"
    )
    with pytest.raises(ValueError) as refusal:
        oq.quantize_oq_streaming(str(source), str(output), 4, quant_format=toy)
    assert str(refusal.value) == (
        "quant format 'toy' encodes with oQe's importance: pass enhanced=True"
    )
    assert not output.exists() and toy.offered == []


@pytest.mark.parametrize(
    "configured,words",
    [
        (
            {"tensors": ("scales", "extra")},
            "quant format 'toy' is configured with tensors ('scales', 'extra') at "
            "contract 1, registered with ('scales',) at 1: the checkpoint would not "
            "load; report it to the plugin",
        ),
        (
            {"contract": 2},
            "quant format 'toy' is configured with tensors ('scales',) at contract 2, "
            "registered with ('scales',) at 1: the checkpoint would not load; report "
            "it to the plugin",
        ),
    ],
)
def test_a_configured_object_that_would_not_load_is_refused(
    source, toy, tmp_path, configured, words
):
    fitted = _Codec()
    for name, value in configured.items():
        setattr(fitted, name, value)
    output = tmp_path / "refused"
    with pytest.raises(ValueError) as refusal:
        oq.quantize_oq_streaming(
            str(source), str(output), 4, enhanced=True, quant_format=fitted
        )
    assert str(refusal.value) == words
    assert not output.exists() and fitted.offered == toy.offered == []


@pytest.mark.parametrize("kind", ["raises", "configures", "returns"])
def test_a_record_the_registered_format_does_not_read_is_refused(
    source, toy, tmp_path, monkeypatch, kind
):
    """Before a weight is read: as the load would refuse it."""

    def accept(record):
        if kind == "raises":
            raise ValueError("no layout 2")
        if kind == "configures":
            toy.levels = (1, 2)
            return None
        return "levels"

    monkeypatch.setattr(toy, "accept", accept)
    words = {
        "raises": "quant format 'toy' refuses the record it gives (no layout 2): "
        "the checkpoint would not load; report it to the plugin",
        "configures": "quant format 'toy''s accept changes the registered format to "
        "read the record: the checkpoint would not load; report it to the plugin",
        "returns": "quant format 'toy''s accept returns 'levels' for the record it "
        "gives: the checkpoint would not load; report it to the plugin",
    }[kind]
    output = tmp_path / "refused"
    with pytest.raises(ValueError) as refusal:
        oq.quantize_oq_streaming(
            str(source), str(output), 4, enhanced=True, quant_format="toy"
        )
    assert str(refusal.value) == words
    assert not output.exists() and toy.offered == []


@pytest.mark.parametrize("loads", ["none", "other"])
def test_a_configured_object_laid_out_otherwise_than_it_loads_stops_the_run(
    source, toy, tmp_path, loads
):
    """The registered format lays out none of a matrix the configured object
    lays out, or the object its accept returns another layout."""
    fitted = _Codec()
    if loads == "none":
        # Though the object its accept returns would lay it out.
        toy.lays_out = False
        toy.accept = lambda record: _Codec()
    else:
        reader = _Codec()
        reader.layout = lambda k, n, bits, gs: {
            "weight": ((n, k), mx.uint8),
            "scales": ((n,), mx.float16),
        }
        toy.accept = lambda record: reader
    with pytest.raises(ValueError) as refusal:
        _quantize(source, tmp_path / "stopped", quant_format=fitted)
    said = str(refusal.value)
    assert said.startswith("quant format 'toy' lays out ") and said.endswith(
        "the checkpoint would not load; report it to the plugin"
    )
    assert fitted.encoded == []


def test_the_contract_page_says_a_format_object_is_taken():
    assert "an object of the class registered for its mode" in _QUOTED
    assert "pass an object of the registered class" in _QUOTED


def test_a_record_a_source_carries_is_not_kept_without_its_layers(
    source, toy, tmp_path, monkeypatch, caplog
):
    """A source written in a format carries its record; an output with no
    layer in a format does not."""
    carried = tmp_path / "carried"
    carried.mkdir()
    for path in source.iterdir():
        (carried / path.name).symlink_to(path)
    config = json.loads((source / "config.json").read_text())
    config[_RECORD] = {"toy": {"plugin": None, "contract": 1, "record": {}}}
    (carried / "config.json").unlink()
    (carried / "config.json").write_text(json.dumps(config))
    _, written, _ = _quantize(carried, tmp_path / "oqe")
    assert _RECORD not in written
    toy.lays_out = False
    _, written, _ = _quantize(carried, tmp_path / "oqe-none", quant_format="toy")
    assert _RECORD not in written


def _cache(path):
    """An imatrix cache's arrays and its metadata, without the batch plan:
    its readings of the memory free at the time change between two
    collections of the same cache."""
    with np.load(path) as arrays:
        kept = {name: arrays[name] for name in arrays.files}
    metadata = json.loads(path.with_name(path.name + ".json").read_text())
    del metadata["collection"]["batch_plan"]
    return kept, metadata


def test_a_calibration_alone_is_the_one_a_run_makes_and_reuses(source, toy, tmp_path):
    """calibrate_oq_streaming writes the imatrix cache that a run with the
    same settings writes, and the source's reference beside it, nothing
    else; that run then reuses it."""
    alone, sampled = tmp_path / "alone" / "imatrix.npz", []
    with pytest.MonkeyPatch.context() as patch:
        _calibration(patch, sampled)
        data = oq.calibrate_oq_streaming(str(source), str(alone), 4, **_SETTINGS)
    assert sampled and not data.reused
    assert sorted(path.name for path in alone.parent.iterdir()) == [
        "imatrix.npz",
        "imatrix.npz.json",
        f"{source.name}.reference.json",
        f"{source.name}.reference.npz",
    ]
    assert sorted(data.entries) == sorted(oq._load_oqe_imatrix(alone).entries)
    # A plain run collects the same cache.
    plain = tmp_path / "plain" / "imatrix.npz"
    _quantize(source, tmp_path / "plain" / "oqe", imatrix_cache_path=str(plain))
    (arrays, metadata), (their_arrays, their_metadata) = _cache(alone), _cache(plain)
    assert sorted(arrays) == sorted(their_arrays)
    assert all(np.array_equal(arrays[name], their_arrays[name]) for name in arrays)
    assert metadata == their_metadata
    # A run with a format reads the calibration, and samples nothing.
    sampled.clear()
    _, _, applied = _quantize(
        source,
        tmp_path / "oqe-toy",
        sampled,
        quant_format="toy",
        imatrix_cache_path=str(alone),
    )
    assert sampled == [] and len(toy.encoded) == len(applied) > 0
    report = json.loads((tmp_path / "oqe-toy" / "oq_imatrix_report.json").read_text())
    assert report["cache_reused"] is True


def test_a_cache_is_written_and_read_at_the_name_given(source, toy, tmp_path):
    """Without .npz too: numpy's would add it; the run then finds it."""
    cache = tmp_path / "cache" / "imatrix"
    with pytest.MonkeyPatch.context() as patch:
        _calibration(patch)
        oq.calibrate_oq_streaming(
            str(source), str(cache), 4, enhanced=True, **_SETTINGS
        )
    assert sorted(path.name for path in cache.parent.iterdir()) == [
        "imatrix",
        "imatrix.json",
        f"{source.name}.reference.json",
        f"{source.name}.reference.npz",
    ]
    output = tmp_path / "oqe-toy"
    _quantize(source, output, quant_format="toy", imatrix_cache_path=str(cache))
    report = json.loads((output / "oq_imatrix_report.json").read_text())
    assert report["cache_reused"] is True


def test_a_run_without_a_checkpoint_refuses_what_names_no_cache(source, toy, tmp_path):
    folder = tmp_path / "folder"
    folder.mkdir()
    for cache, words in (
        ("", "calibrate_oq_streaming writes a cache: name its file"),
        (
            str(folder),
            f"imatrix_cache_path {folder} is a folder: name the cache's file",
        ),
    ):
        with pytest.raises(ValueError) as refusal:
            oq.calibrate_oq_streaming(str(source), cache, 4)
        assert str(refusal.value) == words
    with pytest.raises(ValueError) as refusal:
        oq.quantize_oq_streaming(
            str(source),
            str(tmp_path / "out"),
            4,
            enhanced=True,
            imatrix_cache_path=str(folder),
        )
    assert str(refusal.value) == words
    with pytest.raises(ValueError) as refusal:
        oq.calibrate_oq_streaming(
            str(source), str(tmp_path / "c.npz"), 4, enhanced=False
        )
    assert str(refusal.value) == "a run without a checkpoint is oQe's: enhanced is True"
    with pytest.raises(ValueError) as refusal:
        next(oq.iter_format_offers(str(source), 4, None))
    assert str(refusal.value) == (
        "iter_format_offers offers a quant format's matrices: name one"
    )
    assert list(folder.iterdir()) == [] and not (tmp_path / "out").exists()


@pytest.mark.parametrize("call", ["calibrate", "offers"])
def test_a_refused_run_leaves_no_folder_it_made_for_the_cache(
    source, toy, tmp_path, call
):
    cache = tmp_path / "a" / "b" / "imatrix.npz"
    with pytest.raises(ValueError, match="Invalid oQ level"):
        if call == "calibrate":
            oq.calibrate_oq_streaming(str(source), str(cache), 1)
        else:
            next(
                oq.iter_format_offers(
                    str(source), 1, "toy", imatrix_cache_path=str(cache)
                )
            )
    assert list(tmp_path.iterdir()) == []


def test_the_offers_shapes_take_a_sanitize_s_tensors(source):
    """A sanitize that fell back to an eager run sets tensors on the index:
    new names, and checkpoint names over again."""
    index = oq._LazyTensorIndex(oq._source_weight_files(source))
    name = next(iter(index.keys()))
    index["model.extra.weight"] = mx.zeros((3, 64))
    index[name] = mx.zeros((5, 7))
    shapes = oq._logical_shapes(index)
    assert shapes["model.extra.weight"] == (3, 64) and shapes[name] == (5, 7)
    assert set(shapes) == set(index.keys())


def _offers(source, cache):
    with pytest.MonkeyPatch.context() as patch:
        _calibration(patch)
        return list(
            oq.iter_format_offers(
                str(source), 4, "toy", imatrix_cache_path=cache, **_SETTINGS
            )
        )


@pytest.mark.parametrize("left", [None, "dropped", "packed"])
def test_the_offers_are_the_matrices_a_run_offers_its_format(
    source, toy, tmp_path, monkeypatch, left
):
    """iter_format_offers yields what a run then offers the format, in its
    order: the same matrices, read the same, with the same importance, bits
    and group size; it encodes and writes none. So also when the run drops a
    matrix (an MTP tensor of a run that keeps none) or writes it as its
    source packed it."""
    cache = str(tmp_path / "imatrix.npz")
    if left is not None:
        # The first matrix a run offers, left out of the run.
        first = _offers(source, cache)[0].name
        if left == "dropped":
            mtp = oq._is_mtp_tensor
            monkeypatch.setattr(
                oq, "_is_mtp_tensor", lambda name: name == first or mtp(name)
            )
        else:
            packed = {"bits": 4, "group_size": 64, "mode": "affine"}
            monkeypatch.setattr(
                oq,
                "_kept_packed",
                lambda run, weights, name: packed if name == first else None,
            )
            monkeypatch.setattr(
                oq._DiscoveredPlan,
                "pop_packed",
                lambda self, key: (self.pop(key), mx.zeros((1,))),
            )
    offers = _offers(source, cache)
    assert offers and toy.encoded == []
    if left is not None:
        assert first not in [offer.name for offer in offers]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "imatrix.npz",
        "imatrix.npz.json",
    ]
    _, config, _ = _quantize(
        source, tmp_path / "oqe", quant_format="toy", imatrix_cache_path=cache
    )
    names = [offer.name.removesuffix(".weight") for offer in offers]
    assert names == list(_in_format(config))
    assert len(toy.encoded) == len(offers)
    for offer, (w, importance, bits, group_size) in zip(offers, toy.encoded):
        assert offer.shape == tuple(w.shape)
        assert (offer.bits, offer.group_size) == (bits, group_size)
        # Read as often as asked: the source's tensors stay to be read.
        assert _same({"w": offer.load()}, {"w": w})
        assert _same({"w": offer.load()}, {"w": w})
        assert _same({"i": offer.importance}, {"i": importance})


@pytest.mark.parametrize("named_cache", [True, False])
def test_fitted_widths_match_the_written_quantization_after_memory_pressure(
    source, toy, tmp_path, monkeypatch, named_cache
):
    _calibration(monkeypatch)
    monkeypatch.setenv("OMLX_BASE_PATH", str(tmp_path / "settings"))
    budget = oq._calibration_memory_budget
    pressure = False

    def live_budget(*args, **kwargs):
        return {**budget(*args, **kwargs), "requires_proxy": pressure}

    monkeypatch.setattr(oq, "_calibration_memory_budget", live_budget)
    # The two measurement routes disagree on which layer merits the next bit.
    # Fitting and writing use the actual imatrix, plan, encoder and shards.
    monkeypatch.setattr(oq, "_measure_sensitivity", lambda *a, **kw: {0: 1, 1: 0.01})
    monkeypatch.setattr(
        oq,
        "_measure_sensitivity_from_quantized_model",
        lambda *a, **kw: {0: 0.01, 1: 1},
    )
    settings = dict(
        imatrix_num_samples=4,
        imatrix_seq_length=16,
        stream_calibration=False,
        target_bpw=5.6,
        hard_cap_bpw=5.6,
    )
    if named_cache:
        settings["imatrix_cache_path"] = str(tmp_path / "imatrix.npz")
    offered = {
        offer.name.removesuffix(".weight"): offer.bits
        for offer in oq.iter_format_offers(str(source), 4, toy, **settings)
    }
    assert len(set(offered.values())) > 1
    pressure = True
    output = tmp_path / "oqe"
    writer = _Codec()
    oq.quantize_oq_streaming(
        str(source), str(output), 4, enhanced=True, quant_format=writer, **settings
    )
    written = _in_format(json.loads((output / "config.json").read_text()))
    assert {name: entry["bits"] for name, entry in written.items()} == offered
    assert [bits for _, _, bits, _ in writer.encoded] == list(offered.values())


def test_fitted_widths_respect_an_explicit_sensitivity_source(
    source, toy, tmp_path, monkeypatch
):
    _calibration(monkeypatch)
    cache = str(tmp_path / "imatrix.npz")
    source = shutil.copytree(source, tmp_path / "source")
    first, second = {0: 1, 1: 0.01}, {0: 0.01, 1: 1}
    monkeypatch.setattr(oq, "_measure_sensitivity", lambda *a, **kw: first)
    monkeypatch.setattr(
        oq, "_measure_sensitivity_from_quantized_model", lambda *a, **kw: second
    )

    def widths(**kwargs):
        offers = oq.iter_format_offers(
            str(source),
            4,
            toy,
            imatrix_cache_path=cache,
            stream_calibration=False,
            target_bpw=5.6,
            hard_cap_bpw=5.6,
            imatrix_num_samples=4,
            imatrix_seq_length=16,
            **kwargs,
        )
        return {offer.name: offer.bits for offer in offers}

    fitted = widths()
    expected = widths(sensitivity_map_override=second)
    assert expected != fitted
    assert widths(sensitivity_model_path=str(source)) == expected
    scores = source / "oq_sensitivity_map.json"
    scores.write_text(json.dumps(second))
    assert widths() == expected
    assert widths(sensitivity_map_override=first) == fitted
    scores.unlink()
    assert widths() == fitted


def test_fitted_sensitivity_reuses_only_compatible_measurements(tmp_path):
    metadata = {"source_hash": "first", "calib_data_hash": "corpus"}
    run = oq._OQRun(
        source=tmp_path / "model",
        oq_level=4,
        requested_imatrix_cache=str(tmp_path / "calibration.npz"),
        imatrix_data=SimpleNamespace(metadata=metadata),
        dtype="bfloat16",
        preserve_mtp=False,
        stream_calibration=None,
        imatrix_num_samples=4,
        imatrix_seq_length=16,
        auto_proxy_sensitivity=True,
        trust_remote_code=False,
    )
    calls = []

    def measure():
        calls.append(1)
        return {"0": len(calls)}

    def scores():
        return oq._fitted_sensitivity(run, measure)

    assert scores() == scores() == {"0": 1}
    metadata["source_hash"] = "second"
    assert scores() == {"0": 2}
    metadata["source_hash"] = "first"
    for key, other in {
        "source": tmp_path / "different-model",
        "oq_level": 3,
        "dtype": "float16",
        "preserve_mtp": True,
        "stream_calibration": False,
        "imatrix_num_samples": 8,
        "imatrix_seq_length": 32,
        "auto_proxy_sensitivity": False,
        "trust_remote_code": True,
    }.items():
        previous = run[key]
        run[key] = other
        expected = len(calls) + 1
        assert scores() == scores() == {"0": expected}, key
        run[key] = previous
    assert scores() == {"0": 1}
    metadata["calib_data_hash"] = "new corpus"
    assert scores() == {"0": len(calls)}
    assert scores() != {"0": 1}

    # Two measurements must never publish different priorities for the same
    # fit. Hold the first inside measurement while the second enters the cache.
    from concurrent.futures import ThreadPoolExecutor

    run["oq_level"] = 5
    entered, waiting, release = (threading.Event() for _ in range(3))

    def first():
        entered.set()
        assert release.wait(10)
        return {"0": 0.25}

    def second():
        waiting.set()
        return oq._fitted_sensitivity(run, lambda: {"0": 0.75})

    with ThreadPoolExecutor(max_workers=2) as pool:
        winner = pool.submit(oq._fitted_sensitivity, run, first)
        assert entered.wait(10)
        # A separate reader cannot take even a shared lock while the first
        # measurement is pending.
        import fcntl

        locks = list(tmp_path.glob(".oqe-sensitivity/*.lock"))
        held = 0
        for lock in locks:
            with lock.open() as other:
                try:
                    fcntl.flock(other, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    held += 1
        assert held == 1
        contender = pool.submit(second)
        try:
            assert waiting.wait(10)
            assert not contender.done()
        finally:
            release.set()
        assert winner.result() == contender.result() == {"0": 0.25}
    assert scores() == {"0": 0.25}

    run["oq_level"] = 6
    assert oq._fitted_sensitivity(run, dict) == {}
    assert scores()


def test_the_contract_page_shows_the_offers():
    assert "iter_format_offers(source, 4, fmt" in _QUOTED


def test_a_calibration_alone_refuses_a_writer_of_its_own(tmp_path):
    source = tmp_path / "v41"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
    with pytest.raises(ValueError) as refusal:
        oq.calibrate_oq_streaming(str(source), str(tmp_path / "imatrix.npz"), 4)
    assert str(refusal.value) == (
        "oQ calibrates a deepseek_v41 checkpoint inside a writer of its own, "
        "which takes no quant format"
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["v41"]


def test_the_contract_page_shows_a_calibration_alone():
    assert "calibrate_oq_streaming(source, cache, 4)" in _QUOTED


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
        ({"model_type": "qwen3_vl", "vision_config": {"depth": 1}}, "mlx-vlm"),
    ],
    ids=["llama", "qwen3_vl"],
)
def test_the_walk_builds_the_model_with_the_library_of_the_calibration(
    tmp_path, monkeypatch, config, library
):
    calls, model = _stub_loaders(monkeypatch)
    source = _checkpoint(tmp_path / "source", config)
    walked = oq._module_classes(str(source), config)
    ((loaded, thread, kwargs, given, index),) = calls
    assert loaded == library
    # Lazily, unchecked, on the thread the engines load models on; the
    # model is let go of.
    assert kwargs["lazy"] is True and kwargs["strict"] is False
    assert thread.startswith("mlx-global") and model.closed
    assert given == config and walked["proj"][2] == (16, 64)
    # The view: the source's files but its weights and config, and an index
    # declaring the source's tensors.
    assert index == {"weight_map": {"model.embed_tokens.weight": "model.safetensors"}}


def test_a_text_only_walk_builds_with_the_written_checkpoints_library(
    tmp_path, monkeypatch
):
    calls, _ = _stub_loaders(monkeypatch)
    source = _checkpoint(
        tmp_path / "source", {"model_type": "qwen3_vl", "vision_config": {"depth": 1}}
    )
    # A text-only run writes no vision config.
    oq._module_classes(str(source), {"model_type": "qwen3_vl"})
    assert calls[-1][0] == "mlx-lm"


def test_the_walk_declares_the_source_s_tensors_whatever_its_files(
    tmp_path, monkeypatch
):
    calls, _ = _stub_loaders(monkeypatch)
    config = {"model_type": "llama"}
    source = _checkpoint(tmp_path / "source", config)
    # A file of weights that is a link to nothing declares nothing.
    (source / "extra.safetensors").symlink_to(tmp_path / "nowhere")
    declared = {"weight_map": {"model.embed_tokens.weight": "model.safetensors"}}
    # An index that does not parse, or declares no tensor: the files' headers.
    for text in ('{"weight_map": {}}', '{"weight_map": "ab"}', "[[1]]", "{"):
        (source / "model.safetensors.index.json").write_text(text)
        oq._module_classes(str(source), config)
        assert calls[-1][4] == declared, text
    # The source's own index is not written through.
    assert (source / "model.safetensors.index.json").read_text() == "{"
    index = {"weight_map": {"model.norm.weight": "model.safetensors"}}
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    oq._module_classes(str(source), config)
    assert calls[-1][4] == {"weight_map": {"model.norm.weight": "model.safetensors"}}
    assert json.loads((source / "model.safetensors.index.json").read_text()) == index


def test_one_mlx_executor_whoever_asks_first(monkeypatch):
    import concurrent.futures
    import time

    from omlx import engine_core

    made = []

    class Slow(concurrent.futures.ThreadPoolExecutor):
        def __init__(self, **kwargs):
            # Long enough for the other first caller to ask meanwhile.
            time.sleep(0.2)
            super().__init__(**kwargs)
            made.append(self)

    monkeypatch.setattr(engine_core, "_global_mlx_executor", None)
    monkeypatch.setattr(engine_core.concurrent.futures, "ThreadPoolExecutor", Slow)
    got = []
    threads = [
        threading.Thread(target=lambda: got.append(engine_core.get_mlx_executor()))
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    for executor in made:
        executor.shutdown()
    assert len(made) == 1 and got == made * 2


def test_a_walk_after_the_executor_is_shut_down_runs_on_the_next(monkeypatch):
    import concurrent.futures

    from omlx import engine_core

    gone = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    gone.shutdown()
    live = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="mlx-global"
    )
    executors = iter((gone, live))
    monkeypatch.setattr(engine_core, "get_mlx_executor", lambda: next(executors))
    try:
        assert oq._on_mlx_thread(lambda: 7) == 7
    finally:
        live.shutdown()


def test_the_executor_is_read_under_its_lock(monkeypatch):
    import threading as threads

    from omlx import engine_core

    class Shut:
        """The lock, with a shutdown taking the executor as it is let go."""

        def __init__(self):
            self.lock = threads.Lock()

        def __enter__(self):
            self.lock.acquire()

        def __exit__(self, *raised):
            self.lock.release()
            engine_core._global_mlx_executor = None

    monkeypatch.setattr(engine_core, "_global_mlx_executor", None)
    monkeypatch.setattr(engine_core, "_GLOBAL_MLX_EXECUTOR_LOCK", Shut())
    executor = engine_core.get_mlx_executor()
    assert executor is not None
    executor.shutdown()


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

    for engine, loader, extra in (
        (batched.BatchedEngine, "_load_model_sync", ""),
        (vlm.VLMBatchedEngine, "_load_vlm_sync", ", for_vlm=True"),
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
        # With the arguments the engine gave them before.
        (call,) = (
            node
            for node in ast.walk(nested)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "maybe_apply_pre_load_patches"
        )
        assert ast.unparse(call) == (
            "maybe_apply_pre_load_patches(self._model_name, "
            f"model_settings=self._model_settings{extra})"
        )


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
    # And the records of the run that wrote it, which the walk does not read.
    carried = {
        _RECORD: {"lloyd": {"plugin": None, "contract": 1, "record": {}}},
        _VERIFICATION: {"verified": True, "mode": "lloyd", "verify_version": 0},
    }
    (changed / "config.json").write_text(
        json.dumps({**config, **extra, **carried, "image_token_id": 5})
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
    unlearned = {
        k: v
        for k, v in written.items()
        if k not in (_RECORD, _VERIFICATION, "eos_token_id")
    }
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
    with testing.verified("minimax_m3_vl"):
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
    with (
        testing.verified("minimax_m3_vl"),
        _force_minimax_m3_moe_sanitize_on_load(output),
    ):
        maybe_apply_pre_load_patches(str(output), for_vlm=True)
        model = vlm_utils.load_model(output)
    logits = model.language_model(mx.array([[1, 2, 3, 4]]))
    logits = getattr(logits, "logits", logits)
    assert mx.isfinite(logits).all()
