# SPDX-License-Identifier: Apache-2.0
"""Disarmed Lightning MTP verify rows through Qwen4 attention (Rule 33).

Every verify window is compared with the serial one-row decode steps at the
same positions within the C2 rounding bound (a one-row call is decode and
stays bit-exact), and its cache is left as the serial steps would. Past the
QSA block budget (rank-three positions, below the gathered crossover) a
serial step attends through ``_masked_decode``: one-row block scores, the
one-launch selection kernel and the selected-keys SDPA; a verify window
makes ONE batched selection for the whole window. Below the budget it takes
the fused dense gate (decision 5) or MLX's vector SDPA over each prefix; the
armed ``row_exact_causal_sdpa`` route is never taken. Real attention shapes
(2560 hidden, 24/2 heads of 256, a 4x128 indexer, 2048-token budget),
synthetic 6-bit weights.
"""

from __future__ import annotations

import importlib

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from omlx.custom_kernels.nax import is_nax_available
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
qwen4_exp = importlib.import_module("mlx_vlm.models.qwen4_exp")

from omlx.patches import qwen35_verify_qmm  # noqa: E402
from omlx.patches.mlx_vlm_mtp import qwen35_verify_linear  # noqa: E402
from omlx.patches.qwen35_verify_sdpa_split import (  # noqa: E402
    apply_qwen35_verify_sdpa_split_patch,
)

qwen35_verify_qmm.apply_verify_qmm_patch()
qwen35_verify_linear.apply()
apply_qwen35_verify_sdpa_split_patch()
q35_language = importlib.import_module("mlx_vlm.models.qwen3_5.language")
q35_verifier = importlib.import_module("mlx_vlm.models.qwen3_5.speculative_verifier")


def _float_cache_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
    """mlx-vlm's SDPA helper for a float KV cache."""
    return mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale, mask=mask, sinks=sinks
    )


@pytest.fixture(autouse=True, scope="module")
def _stock_sdpa_helper():
    # Other suites install SDPA wrappers around fakes into these module
    # globals and leave them behind; pin the stock float-cache behavior.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(q35_language, "scaled_dot_product_attention", _float_cache_sdpa)
        patch.setattr(
            q35_verifier,
            "scaled_dot_product_attention",
            _float_cache_sdpa,
            raising=False,
        )
        yield


def _text_config():
    return qwen4_exp.TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=2560,
        num_hidden_layers=4,
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=16,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        max_position_embeddings=262144,
        hc_count=2,
        hc_lowrank=8,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        ple_layer_ids=[],
        ple_embed_dim=32,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        eos_token_id=1,
        rope_parameters={
            "type": "default",
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000,
        },
    )


_CONFIG = _text_config()


def _attention(seed: int):
    mx.random.seed(seed)
    attn = language.Qwen4ExpAttention(_CONFIG)
    for _, module in attn.named_modules():
        if isinstance(module, nn.Linear):
            fan_in = module.weight.shape[1]
            module.weight = (mx.random.normal(module.weight.shape) * fan_in**-0.5).astype(
                mx.bfloat16
            )
    for norm in (attn.q_norm, attn.k_norm, attn.indexer.q_layernorm, attn.indexer.k_layernorm):
        norm.weight = mx.random.normal(norm.weight.shape) * 0.1
    nn.quantize(attn, group_size=64, bits=6, class_predicate=lambda _, m: isinstance(m, nn.Linear))
    for _, module in attn.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            module.scales = module.scales.astype(mx.bfloat16)
            module.biases = module.biases.astype(mx.bfloat16)
    mx.eval(attn.parameters())
    return attn


def _positions(start: int, length: int) -> mx.array:
    """The served rank-three text positions below the gathered crossover."""
    seq = mx.arange(start, start + length, dtype=mx.int32)[None]
    return mx.broadcast_to(seq[None], (3, 1, length))


def _prefill(attn, tokens: int, seed: int, chunk: int = 4096):
    cache = language.QSAKVCache()
    mx.random.seed(seed)
    done = 0
    while done < tokens:
        size = min(chunk, tokens - done)
        x = mx.random.normal((1, size, _CONFIG.hidden_size)).astype(mx.bfloat16)
        attn(x, mask="causal", cache=cache, position_ids=_positions(done, size))
        mx.eval([a for a in cache.state if a is not None])
        done += size
    return cache


def _copy_cache(cache):
    """``cache`` with its own array objects (cache writes update arrays in place)."""
    copy = language.QSAKVCache.__new__(language.QSAKVCache)
    copy.__dict__.update(
        (name, mx.array(value) if isinstance(value, mx.array) else value)
        for name, value in vars(cache).items()
    )
    return copy


@pytest.fixture(scope="module")
def shared_prefill(_stock_sdpa_helper):
    """``(attn, cache)`` for a context: one prefill per context, a copy per case.

    Trimming a verify window restores the offsets, not the position rank or
    buffer capacities, so every case starts from a copy of the prefill.
    """
    attn = _attention(seed=3)
    prefills = {}

    def prefilled(context):
        if context not in prefills:
            prefills[context] = _prefill(attn, context, seed=4)
        return attn, _copy_cache(prefills[context])

    return prefilled


def _verify(attn, cache, x):
    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=False)
    try:
        out = attn(
            x,
            mask="causal",
            cache=cache,
            position_ids=_positions(cache.offset, x.shape[1]),
            target_verify=True,
        )
        mx.eval(out)
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)
    return out


def _serial(attn, cache, x):
    rows = []
    for row in range(x.shape[1]):
        out = attn(
            x[:, row : row + 1],
            mask=None,
            cache=cache,
            position_ids=_positions(cache.offset, 1),
        )
        mx.eval(out)
        rows.append(out)
    return mx.concatenate(rows, axis=1)


def _state(cache):
    arrays = [
        cache.keys[..., : cache.offset, :],
        cache.values[..., : cache.offset, :],
        cache.index_keys,
        cache.index_position_ids,
    ]
    copies = [mx.array(a) for a in arrays]
    mx.eval(copies)
    return copies


def _record_selection(monkeypatch):
    """Every one-row selection (FP32 head scores, key count, token mask)."""
    calls = []
    original = language.Qwen4ExpQSAIndexer.aligned_row_mask

    def recording(self, scores, key_len):
        mask = original(self, scores, key_len)
        mx.eval(scores, mask)
        calls.append((mx.array(scores), key_len, mx.array(mask)))
        return mask

    monkeypatch.setattr(language.Qwen4ExpQSAIndexer, "aligned_row_mask", recording)
    return calls


def _bit_equal(a: mx.array, b: mx.array) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and mx.array_equal(a, b).item()


def _close(want, got, tolerance=2e-2):
    """Rule 33: a disarmed verify window takes the batched attention route,
    so it differs from serial decode by rounding (the C2-style relative
    bound), not bit for bit."""
    if want.dtype != got.dtype or want.shape != got.shape:
        return False
    left = want.astype(mx.float32)
    right = got.astype(mx.float32)
    difference = float(mx.sqrt(mx.sum((left - right) ** 2)).item())
    scale = float(mx.sqrt(mx.sum(left * left)).item())
    return difference <= tolerance * max(scale, 1e-12)


def _check_selection_masks(
    batched_mask, serial_selections, rows, selected_tokens, nax_available
):
    """A batched window's mask must match each serial row's on that row's
    columns and select nothing past them (a causal leak). The one accepted
    difference is MLX 0.32.5's NAX ``qmm_t_nax`` at ``rows >= 7``: a BF16 ULP
    on ``index_qk_proj`` can flip a few boundary blocks at the top-k cutoff."""
    mismatched = 0
    padded_serial = []
    key_len = int(batched_mask.shape[-1])
    for row_idx, (_, _, serial_mask) in enumerate(serial_selections):
        width = int(serial_mask.shape[-1])
        leaked = int(mx.sum(batched_mask[0, 0, row_idx, width:]).item())
        assert leaked == 0, (row_idx, width, key_len, leaked)
        mismatched += int(
            mx.sum(batched_mask[0, 0, row_idx, :width] != serial_mask[0, 0, 0, :]).item()
        )
        padded_serial.append(
            mx.pad(serial_mask, [(0, 0), (0, 0), (0, 0), (0, key_len - width)])
        )
    if mismatched:
        assert nax_available and rows >= 7, (rows, mismatched)
        assert mismatched <= 0.04 * selected_tokens, (mismatched, selected_tokens)
    return mismatched, padded_serial


# 2060: the window starts just past the sparse crossover; 16382: its rows
# straddle MLX's two-pass partition switch at 16384 keys (the multi-row MLX
# path gave row 0 of a two-row window the 16384-key plan); 24000: the served
# 24K case.
@pytest.mark.parametrize("context", [2060, 16382, 24000])
@pytest.mark.parametrize("rows", [2, 3, 8, 16])
def test_masked_verify_rows_equal_serial_decode_steps(
    shared_prefill, monkeypatch, context, rows
):
    attn, cache = shared_prefill(context)
    selections = _record_selection(monkeypatch)
    batched = []
    selection_mask = language.Qwen4ExpQSAIndexer._selection_mask

    def counted(self, *args, **keywords):
        mask = selection_mask(self, *args, **keywords)
        mx.eval(mask)
        batched.append(mx.array(mask))
        return mask

    monkeypatch.setattr(language.Qwen4ExpQSAIndexer, "_selection_mask", counted)
    for seed in (11, 12):
        mx.random.seed(seed + rows)
        x = (mx.random.normal((1, rows, _CONFIG.hidden_size)) * 0.5).astype(mx.bfloat16)

        batched.clear()
        verified = _verify(attn, cache, x)
        verified_state = _state(cache)
        verified_selections = selections[:]
        selections.clear()
        batched_selections = batched[:]
        batched.clear()
        cache.trim(rows)

        serial = _serial(attn, cache, x)
        serial_state = _state(cache)
        serial_selections = selections[:]
        selections.clear()
        cache.trim(rows)

        for got, want in zip(verified_state, serial_state):
            assert _close(got, want)
        # Rule 33: the disarmed window selects once for the batch; the
        # per-row selection hook (seq_len == 1 only) never runs, and score
        # parity against serial decode is the C4e KL bar's job.
        assert verified_selections == []
        assert len(batched_selections) == 1, batched_selections

        mismatched, padded_serial = _check_selection_masks(
            batched_selections[0],
            serial_selections,
            rows,
            rows * attn.indexer.block_topk * attn.indexer.compress_ratio,
            is_nax_available(),
        )
        if mismatched == 0:
            assert _close(verified, serial)
        else:
            # >=98% block overlap, and the attention path within 2e-2 under
            # identical selection masks.
            stacked = mx.concatenate(padded_serial, axis=2)
            with monkeypatch.context() as fixed:
                fixed.setattr(
                    language.Qwen4ExpQSAIndexer,
                    "_selection_mask",
                    lambda self, *a, _stacked=stacked, **kw: _stacked,
                )
                same_mask = _verify(attn, cache, x)
                cache.trim(rows)
            assert _close(same_mask, serial)


def _selection_mask_of(rows, key_len, selected):
    mask = np.zeros((1, 1, rows, key_len), dtype=bool)
    for row, columns in enumerate(selected):
        mask[0, 0, row, columns] = True
    return mx.array(mask)


def test_selection_mask_check_rejects_short_row_mismatch_and_future_leak():
    """The NAX rows>=7 tolerance covers only the NAX route's boundary flips:
    an exact route's mismatch or a bit past a row's own width must fail."""
    serial = [
        (None, None, _selection_mask_of(1, 2, [[0, 1]])),
        (None, None, _selection_mask_of(1, 4, [[0, 2, 3]])),
    ]
    exact = _selection_mask_of(2, 5, [[0, 1], [0, 2, 3]])
    assert _check_selection_masks(exact, serial, 2, 100, True)[0] == 0
    assert _check_selection_masks(exact, serial, 8, 100, True)[0] == 0

    flipped = _selection_mask_of(2, 5, [[1], [0, 2, 3]])
    with pytest.raises(AssertionError):
        _check_selection_masks(flipped, serial, 2, 100, False)
    with pytest.raises(AssertionError):
        _check_selection_masks(flipped, serial, 2, 100, True)
    with pytest.raises(AssertionError):
        _check_selection_masks(flipped, serial, 8, 100, False)
    assert _check_selection_masks(flipped, serial, 8, 100, True)[0] == 1

    leaky = _selection_mask_of(2, 5, [[0, 1, 2], [0, 2, 3]])
    with pytest.raises(AssertionError):
        _check_selection_masks(leaky, serial, 8, 100, True)


def test_masked_verify_rollback_then_decode_matches_serial():
    """Accept one draft of a four-row window, roll the rest back, decode on."""
    attn = _attention(seed=5)
    # Past the block budget and two tokens into a block: the accepted rows
    # complete it and the rollback lands on its boundary.
    reference = _prefill(attn, 2062, seed=6)
    cache = _prefill(attn, 2062, seed=6)
    mx.random.seed(21)
    window = (mx.random.normal((1, 4, _CONFIG.hidden_size)) * 0.5).astype(mx.bfloat16)
    follow = (mx.random.normal((1, 3, _CONFIG.hidden_size)) * 0.5).astype(mx.bfloat16)

    verified = _verify(attn, cache, window)
    cache.trim(2)  # rows 2 and 3 rejected
    resumed = _serial(attn, cache, follow)

    expected = _serial(attn, reference, window[:, :2])
    expected_resumed = _serial(attn, reference, follow)
    assert _close(verified[:, :2], expected)
    assert _close(resumed, expected_resumed)
    for got, want in zip(_state(cache), _state(reference)):
        assert _close(got, want)


# Below the block budget: 700 keys stays on MLX's one-pass vector kernel;
# windows at 1020/1022 put rows on both sides of its two-pass switch at 1024
# keys, which a shared multi-row call would hide.
@pytest.mark.parametrize("context", [700, 1020, 1022])
@pytest.mark.parametrize("rows", [2, 3, 4, 8, 16])
def test_dense_verify_rows_equal_serial_decode_steps(context, rows, monkeypatch):
    import omlx.patches.qwen35_verify_sdpa_split as sdpa_split

    def no_row_exact(*args, **kwargs):
        pytest.fail("a disarmed verify window ran the row-exact SDPA")

    monkeypatch.setattr(sdpa_split, "row_exact_causal_sdpa", no_row_exact)
    attn = _attention(seed=8)
    cache = _prefill(attn, context, seed=9)
    for seed in (31, 32):
        mx.random.seed(seed + rows)
        x = (mx.random.normal((1, rows, _CONFIG.hidden_size)) * 0.5).astype(mx.bfloat16)
        verified = _verify(attn, cache, x)
        verified_state = _state(cache)
        cache.trim(rows)
        serial = _serial(attn, cache, x)
        serial_state = _state(cache)
        cache.trim(rows)
        assert _close(verified, serial)
        for got, want in zip(verified_state, serial_state):
            assert _close(got, want)
