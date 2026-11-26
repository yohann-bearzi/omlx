# SPDX-License-Identifier: Apache-2.0
"""Regression tests for O11: zero host syncs on ragged batched Qwen4-Exp verify.

Pins:
1. ``_gathered_batch_paddings`` checks the QSA crossover upper bound
   ``(width + length) // ratio <= topk`` before any host read of
   ``cache.left_padding``.
2. ``BatchKVCache`` maintains a host ``list[int]`` shadow of ``left_padding``
   across init, ``prepare``, ``finalize``, ``filter``, ``extend``, ``merge``,
   ``extract``, and ``QSAKVCache.to_batch``, keeping ``left_padding``
   materialized on the host so ragged verify rollbacks never leave an
   unevaluated GPU array that drains the pipeline on the next step.
3. External direct assignments, in-place ``+=``/``-=``, and ``state`` restores
   invalidate the shadow so it can never go stale.
4. Outputs remain bitwise identical to the unshadowed / dense reference both
   below and above the QSA crossover, including across concurrent threads.
"""

from __future__ import annotations

import copy
import importlib
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models import cache as lm_cache
from mlx_vlm.models import cache as vlm_cache

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from omlx.patches.vlm_batch_kv_capacity import apply_batch_kv_capacity_patch

apply_batch_kv_capacity_patch()
compat.apply_mlx_vlm_qwen4_exp_compat_patch()

language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
qwen4_cache = importlib.import_module("mlx_vlm.models.qwen4_exp.cache")


def _bits(arr: mx.array) -> np.ndarray:
    if arr.dtype == mx.bfloat16:
        arr = arr.view(mx.uint16)
    return np.array(arr)


def _assert_bitwise(a: mx.array, b: mx.array) -> None:
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    np.testing.assert_array_equal(_bits(a), _bits(b))


def _tiny_config(*, budget: int = 32, ratio: int = 4, attn_layers: int = 4):
    from mlx_vlm.models.qwen4_exp import TextConfig

    return TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=32,
        num_hidden_layers=attn_layers * 2,
        num_attention_heads=4,
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
        num_key_value_heads=2,
        max_position_embeddings=256,
        hc_count=2,
        hc_lowrank=8,
        head_dim=8,
        layer_types=["linear_attention", "qwen_sparse_attention"] * attn_layers,
        ple_layer_ids=[],
        ple_embed_dim=32,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=budget,
        indexer_compress_ratio=ratio,
        eos_token_id=1,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 10_000,
            "partial_rotary_factor": 1.0,
        },
    )


def _root_config():
    return SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=58,
    )


def _reset_rope(model) -> None:
    model._position_ids = None
    model._rope_deltas = None


def _build_batch_model(
    *,
    budget: int = 32,
    ratio: int = 4,
    attn_layers: int = 4,
    prefixes: tuple[int, int] = (6, 4),
    seed: int = 11,
):
    mx.random.seed(seed)
    cfg = _tiny_config(budget=budget, ratio=ratio, attn_layers=attn_layers)
    model = language.LanguageModel(cfg, _root_config())
    mx.eval(model.parameters())

    c0 = model.make_cache()
    c1 = model.make_cache()
    p0 = mx.arange(1, prefixes[0] + 1, dtype=mx.int32)[None]
    p1 = mx.arange(1, prefixes[1] + 1, dtype=mx.int32)[None]
    mx.eval(model(p0, cache=c0).logits)
    _reset_rope(model)
    mx.eval(model(p1, cache=c1).logits)
    _reset_rope(model)

    caches = [a.merge([a, b]) for a, b in zip(c0, c1)]
    mx.eval([c.state for c in caches])
    return model, caches


def _attn_caches(caches):
    return [c for c in caches if isinstance(c, language.BatchQSAKVCache)]


def _make_canary(tag: float) -> mx.array:
    # 16 MB unevaluated GPU array; evaluating it jumps active memory by 16 MB.
    return mx.ones((4 * 1024 * 1024,), dtype=mx.float32) + tag


def test_ragged_verify_below_crossover_performs_zero_left_padding_syncs(monkeypatch):
    """Below the QSA crossover, a ragged batch=2 verify must not read or sync
    ``left_padding`` even when every layer's ``left_padding`` is an unevaluated
    GPU array."""
    model, caches = _build_batch_model(budget=32, ratio=4, prefixes=(6, 4), seed=17)
    ref_caches = copy.deepcopy(caches)

    inp1 = mx.array([[11, 12], [13, 14]], dtype=mx.int32)
    inp2 = mx.array([[15, 16], [17, 18]], dtype=mx.int32)

    out1 = model(inp1, cache=caches, return_hidden=True)
    mx.eval(out1.logits)
    model.rollback_speculative_cache(caches, out1.gdn_states, [1, 0], 2)
    _reset_rope(model)

    ref_out1 = model(inp1, cache=ref_caches, return_hidden=True)
    mx.eval(ref_out1.logits)
    model.rollback_speculative_cache(ref_caches, ref_out1.gdn_states, [1, 0], 2)
    mx.eval([c.state for c in ref_caches])
    _reset_rope(model)

    # Plant a 16 MB unevaluated GPU canary on every attention layer's
    # left_padding. If any layer calls .tolist() before the crossover check,
    # active memory jumps by >= 16 MB per layer during model() build.
    canaries = []
    for idx, cache in enumerate(_attn_caches(caches)):
        canary = _make_canary(float(idx + 1))
        canaries.append(canary)
        cache.kv_cache.left_padding = mx.depends(cache.kv_cache.left_padding, canary)

    monkeypatch.setattr(language, "_EAGER_DISPATCH", False)
    mx.synchronize()
    mem_before = mx.get_active_memory()
    out2 = model(inp2, cache=caches, return_hidden=True)
    mem_after_build = mx.get_active_memory()

    # Zero canaries evaluated during forward graph construction.
    assert mem_after_build - mem_before < canaries[0].nbytes // 2, (
        f"left_padding synced to host during below-crossover ragged verify "
        f"(active memory jumped by {(mem_after_build - mem_before) / 1e6:.2f} MB)"
    )

    _reset_rope(model)
    ref_out2 = model(inp2, cache=ref_caches, return_hidden=True)
    mx.eval(
        out2.logits, out2.hidden_states[0], ref_out2.logits, ref_out2.hidden_states[0]
    )
    _assert_bitwise(out2.logits, ref_out2.logits)
    _assert_bitwise(out2.hidden_states[0], ref_out2.hidden_states[0])


@pytest.mark.parametrize("gathered_disabled", [False, True])
def test_ragged_verify_above_crossover_uses_host_shadow_without_gpu_sync(
    monkeypatch, gathered_disabled
):
    """Above the QSA crossover (both gathered batch and dense from_projected),
    a ragged rollback keeps the host shadow live and never syncs the GPU
    array during the next verify forward."""
    # budget=8, ratio=2 -> crossover at > 8 tokens; prefixes (14, 11) are above.
    model, caches = _build_batch_model(budget=8, ratio=2, prefixes=(14, 11), seed=23)
    ref_caches = copy.deepcopy(caches)

    inp1 = mx.array([[11, 12], [13, 14]], dtype=mx.int32)
    inp2 = mx.array([[15, 16], [17, 18]], dtype=mx.int32)

    # First verify + ragged rollback ([0, 1] rolls row 0 right by 1).
    out1 = model(inp1, cache=caches, return_hidden=True)
    mx.eval(out1.logits)
    model.rollback_speculative_cache(caches, out1.gdn_states, [0, 1], 2)
    _reset_rope(model)

    ref_out1 = model(inp1, cache=ref_caches, return_hidden=True)
    mx.eval(ref_out1.logits)
    model.rollback_speculative_cache(ref_caches, ref_out1.gdn_states, [0, 1], 2)
    mx.eval([c.state for c in ref_caches])
    _reset_rope(model)

    # Every attention cache must have an up-to-date host shadow after rollback
    # (initial left_padding [0, 3] + right_padding [1, 0] -> [1, 3]).
    canaries = []
    for idx, cache in enumerate(_attn_caches(caches)):
        assert cache.left_padding_list() == [1, 3]
        assert cache.kv_cache._left_padding_host == [1, 3]
        # Attach a 16 MB lazy canary directly to the underlying array while
        # keeping the host shadow intact: any .tolist()/.item() on the GPU
        # array will immediately evaluate the canary.
        canary = _make_canary(float(idx + 10))
        canaries.append(canary)
        cache.kv_cache._left_padding = mx.depends(cache.kv_cache._left_padding, canary)

    # Clear the reference caches' host shadow so the reference exercises the
    # array fallback path, proving bitwise parity between shadow and fallback.
    for cache in _attn_caches(ref_caches):
        cache.kv_cache._left_padding_host = None

    monkeypatch.setattr(language, "_EAGER_DISPATCH", False)
    monkeypatch.setattr(language, "_GATHERED_BATCH_DISABLED", gathered_disabled)
    mx.synchronize()
    mem_before = mx.get_active_memory()
    out2 = model(inp2, cache=caches, return_hidden=True)
    mem_after_build = mx.get_active_memory()

    assert mem_after_build - mem_before < canaries[0].nbytes // 2, (
        f"above-crossover ragged verify synced left_padding to host "
        f"(gathered_disabled={gathered_disabled}, "
        f"delta={(mem_after_build - mem_before) / 1e6:.2f} MB)"
    )

    _reset_rope(model)
    ref_out2 = model(inp2, cache=ref_caches, return_hidden=True)
    mx.eval(
        out2.logits, out2.hidden_states[0], ref_out2.logits, ref_out2.hidden_states[0]
    )
    _assert_bitwise(out2.logits, ref_out2.logits)
    _assert_bitwise(out2.hidden_states[0], ref_out2.hidden_states[0])


def test_finalize_materializes_left_padding_on_host_not_lazy_gpu_add():
    """After prepare + finalize, reading cache.left_padding.tolist() directly
    (as external callers like mlx-vlm decode mask builders do) must not
    evaluate a lazy GPU chain attached to the previous left_padding."""
    model, caches = _build_batch_model(budget=32, ratio=4, prefixes=(6, 4), seed=29)
    attn_cache = _attn_caches(caches)[0]
    kv = attn_cache.kv_cache
    assert kv._left_padding_host == [0, 2]

    # Attach a 16 MB lazy canary to the pre-rollback array while keeping the
    # shadow intact. If finalize() computes `left_padding += padding` on the
    # GPU instead of rebuilding left_padding from the host shadow, calling
    # `.tolist()` after finalize() will trigger the canary.
    canary = _make_canary(99.0)
    kv._left_padding = mx.depends(kv._left_padding, canary)

    attn_cache.prepare(right_padding=[1, 0])
    attn_cache.finalize()

    mx.synchronize()
    mem_before = mx.get_active_memory()
    pads = attn_cache.left_padding.tolist()
    mem_after = mx.get_active_memory()

    assert pads == [1, 2]
    assert attn_cache.left_padding_list() == [1, 2]
    assert mem_after - mem_before < canary.nbytes // 2, (
        f"finalize() left a lazy GPU chain on left_padding "
        f"(delta={(mem_after - mem_before) / 1e6:.2f} MB)"
    )


def test_batched_decode_propagates_left_padding_to_all_attention_layers(monkeypatch):
    """Standard batched decode (length=1) after ragged rollback must not sync
    any attention layer's left_padding and must set _qwen3_5_decode_left_padding
    uniformly across all full-attention caches."""
    model, caches = _build_batch_model(budget=32, ratio=4, prefixes=(6, 4), seed=31)
    ref_caches = copy.deepcopy(caches)

    inp = mx.array([[11, 12], [13, 14]], dtype=mx.int32)
    out1 = model(inp, cache=caches, return_hidden=True)
    mx.eval(out1.logits)
    model.rollback_speculative_cache(caches, out1.gdn_states, [1, 0], 2)
    _reset_rope(model)

    ref_out1 = model(inp, cache=ref_caches, return_hidden=True)
    mx.eval(ref_out1.logits)
    model.rollback_speculative_cache(ref_caches, ref_out1.gdn_states, [1, 0], 2)
    mx.eval([c.state for c in ref_caches])
    _reset_rope(model)

    tok = mx.array([[19], [20]], dtype=mx.int32)
    out_dec = model(tok, cache=caches)
    _reset_rope(model)
    ref_dec = model(tok, cache=ref_caches)
    mx.eval(out_dec.logits, ref_dec.logits)
    _assert_bitwise(out_dec.logits, ref_dec.logits)

    for cache in _attn_caches(caches):
        assert getattr(cache, "_qwen3_5_decode_left_padding", None) == [0, 3]

    # A subsequent verify step must clear _qwen3_5_decode_left_padding everywhere.
    _reset_rope(model)
    out_ver = model(inp, cache=caches, return_hidden=True)
    mx.eval(out_ver.logits)
    for cache in _attn_caches(caches):
        assert not hasattr(cache, "_qwen3_5_decode_left_padding")


@pytest.mark.parametrize(
    "cache_mod",
    [vlm_cache, lm_cache, qwen4_cache],
    ids=["vlm", "lm", "qwen4_exp"],
)
def test_batch_kv_cache_left_padding_host_shadow_lifecycle_and_invalidation(cache_mod):
    """Host shadow stays exact through init, prepare, finalize, filter, extend,
    merge, extract, deepcopy, and invalidates cleanly on direct mutation."""
    cls = cache_mod.BatchKVCache

    # Init + prepare(left_padding=...) on empty cache
    c = cls([1, 0])
    assert c.left_padding_list() == [1, 0]
    c.prepare(left_padding=[2, 1])
    assert c.left_padding_list() == [3, 1]
    assert c.left_padding.tolist() == [3, 1]

    # Mutating returned list must not corrupt internal shadow.
    returned = c.left_padding_list()
    returned[0] = 999
    assert c.left_padding_list() == [3, 1]

    # Update + prepare(right_padding=...) + finalize()
    k = mx.ones((2, 2, 6, 8), dtype=mx.bfloat16)
    v = mx.ones((2, 2, 6, 8), dtype=mx.bfloat16)
    c.update_and_fetch(k, v)
    c.prepare(right_padding=[2, 0])
    c.finalize()
    assert c.left_padding_list() == [5, 1]
    assert c.left_padding.tolist() == [5, 1]

    # Deepcopy keeps independent shadows.
    cloned = copy.deepcopy(c)
    cloned.prepare(right_padding=[0, 1])
    cloned.finalize()
    assert cloned.left_padding_list() == [5, 2]
    assert c.left_padding_list() == [5, 1]

    # Filter drops common min_left_pad (1) -> [4, 0] or single row [0].
    c_filt = copy.deepcopy(c)
    c_filt.filter([0, 1])
    assert c_filt.left_padding_list() == [4, 0]
    assert c_filt.left_padding.tolist() == [4, 0]
    assert c_filt.left_padding.dtype == mx.int32

    # Filter to empty preserves dtype and empty list.
    c_empty = copy.deepcopy(c)
    c_empty.filter(mx.array([], dtype=mx.int32))
    assert c_empty.left_padding_list() == []
    assert c_empty.left_padding.shape == (0,)
    assert c_empty.left_padding.dtype == mx.int32

    # Extend across unequal widths updates shadow cleanly.
    other = cls([0])
    other.update_and_fetch(
        mx.ones((1, 2, 9, 8), dtype=mx.bfloat16),
        mx.ones((1, 2, 9, 8), dtype=mx.bfloat16),
    )
    c_ext = copy.deepcopy(c)
    c_ext.extend(other)
    # c had _idx=6, pads=[5, 1]; other has _idx=9, pads=[0] -> shift=3, left=0 -> [8, 4, 0]
    assert c_ext.left_padding_list() == [8, 4, 0]
    assert c_ext.left_padding.tolist() == [8, 4, 0]

    # Extract uses host shadow without syncing the GPU array.
    ext_canary = _make_canary(33.0)
    c_ext._left_padding = mx.depends(c_ext._left_padding, ext_canary)
    mx.synchronize()
    em0 = mx.get_active_memory()
    extracted = c_ext.extract(1)
    em1 = mx.get_active_memory()
    assert em1 - em0 < ext_canary.nbytes // 2
    assert extracted.offset == 9 - 4

    # Direct assignment invalidates shadow (never stale).
    c_ext.left_padding = mx.array([1, 2, 3], dtype=mx.int32)
    assert c_ext._left_padding_host is None
    assert c_ext.left_padding_list() == [1, 2, 3]

    # In-place += invalidates shadow via property setter (never stale).
    c_ext.left_padding += mx.array([2, 0, 1], dtype=mx.int32)
    assert c_ext._left_padding_host is None
    assert c_ext.left_padding_list() == [3, 2, 4]

    # State setter invalidates shadow.
    st = c_ext.state
    restored = cls([0, 0, 0])
    restored.state = st
    assert restored._left_padding_host is None
    assert restored.left_padding_list() == [3, 2, 4]

    # Manual _right_padding overwrite without prepare() must not read stale host list.
    c_manual = copy.deepcopy(c)
    c_manual.prepare(right_padding=[1, 0])
    c_manual._right_padding = mx.array([0, 2], dtype=mx.int32)
    c_manual.finalize()
    assert c_manual.left_padding_list() == [5, 3]
    assert c_manual.left_padding.tolist() == [5, 3]

    # Empty-empty extend keeps shadow intact.
    e1, e2 = cls([1, 0]), cls([2])
    e1.extend(e2)
    assert e1._left_padding_host == [1, 0, 2]
    assert e1.left_padding_list() == [1, 0, 2]
    assert e1.left_padding.tolist() == [1, 0, 2]

    # In-place extend into spare rows (filter leaves spare row -> extend reuses it).
    c_spare = cls([0, 1, 2])
    c_spare.update_and_fetch(
        mx.ones((3, 2, 6, 8), dtype=mx.bfloat16),
        mx.ones((3, 2, 6, 8), dtype=mx.bfloat16),
    )
    c_spare.filter([0, 2])  # drops min=0, keeps rows [0, 2] -> pads [0, 2], 1 spare row
    assert c_spare._rows == 2
    assert c_spare.left_padding_list() == [0, 2]
    donor = cls([1])
    donor.update_and_fetch(
        mx.ones((1, 2, 4, 8), dtype=mx.bfloat16),
        mx.ones((1, 2, 4, 8), dtype=mx.bfloat16),
    )
    c_spare.extend(donor)  # _extend_in_place: shift=0, left=2 -> [0, 2, 3]
    assert c_spare.left_padding_list() == [0, 2, 3]
    assert c_spare.left_padding.tolist() == [0, 2, 3]

    # Stock concatenate fallback (mismatched KV dtypes) keeps shadow intact.
    c_f16 = cls([1])
    c_f16.update_and_fetch(
        mx.ones((1, 2, 4, 8), dtype=mx.float16),
        mx.ones((1, 2, 4, 8), dtype=mx.float16),
    )
    c_bf16 = cls([0])
    c_bf16.update_and_fetch(
        mx.ones((1, 2, 6, 8), dtype=mx.bfloat16),
        mx.ones((1, 2, 6, 8), dtype=mx.bfloat16),
    )
    c_f16.extend(c_bf16)  # shift=2, left=0 -> [3, 0]
    assert c_f16.left_padding_list() == [3, 0]
    assert c_f16.left_padding.tolist() == [3, 0]

    # Fallback when left_padding is non-integer / array-initialized.
    c_arr = cls(mx.array([2, 1], dtype=mx.int32))
    assert c_arr._left_padding_host is None
    assert c_arr.left_padding_list() == [2, 1]
    assert c_arr._left_padding_host == [2, 1]

    # Pre-existing instance compatibility (raw __dict__["left_padding"]).
    c_pre = cls([0, 0])
    del c_pre.__dict__["_left_padding"]
    c_pre.__dict__["left_padding"] = mx.array([4, 1], dtype=mx.int32)
    c_pre._left_padding_host = None
    assert c_pre.left_padding_list() == [4, 1]
    c_pre.left_padding = mx.array([2, 0], dtype=mx.int32)
    assert "left_padding" not in c_pre.__dict__
    assert c_pre.left_padding_list() == [2, 0]


def test_batch_qsa_finalize_avoids_right_padding_tolist_and_handles_plain_kv():
    """BatchQSAKVCache.finalize uses host right_padding without syncing, and
    left_padding_list falls back cleanly on a plain kv_cache stub."""
    c = language.BatchQSAKVCache([0, 0])
    c.update_and_fetch(
        mx.ones((2, 2, 6, 8), dtype=mx.bfloat16),
        mx.ones((2, 2, 6, 8), dtype=mx.bfloat16),
    )
    c.update_indexer(
        mx.ones((2, 6, 8), dtype=mx.bfloat16),
        mx.broadcast_to(mx.arange(6, dtype=mx.int32)[None], (2, 6)),
    )
    mx.eval(c.keys, c.values, c.index_keys, c.index_position_ids)
    c.prepare(right_padding=[2, 0])
    assert c.kv_cache._right_padding_host == (c.kv_cache._right_padding, [2, 0])
    canary = _make_canary(55.0)
    # Plant canary on _right_padding while preserving _right_padding_host ref.
    rp = mx.depends(c.kv_cache._right_padding, canary)
    c.kv_cache._right_padding = rp
    c.kv_cache._right_padding_host = (rp, [2, 0])
    mx.synchronize()
    m0 = mx.get_active_memory()
    c.finalize()
    m1 = mx.get_active_memory()
    assert m1 - m0 < canary.nbytes // 2
    assert c.left_padding_list() == [2, 0]

    # BatchQSAKVCache.extract uses host shadow without syncing left_padding.
    ext_canary = _make_canary(56.0)
    c.kv_cache._left_padding = mx.depends(c.kv_cache._left_padding, ext_canary)
    mx.synchronize()
    em0 = mx.get_active_memory()
    row0 = c.extract(0)
    em1 = mx.get_active_memory()
    assert em1 - em0 < ext_canary.nbytes // 2
    assert row0.offset == 4

    # Plain kv_cache stub without left_padding_list falls back to .tolist().
    c_stub = language.BatchQSAKVCache([0, 0])
    c_stub.kv_cache = SimpleNamespace(left_padding=mx.array([3, 1], dtype=mx.int32))
    assert c_stub.left_padding_list() == [3, 1]


def test_qsa_to_batch_and_extend_preserves_host_shadow():
    """QSAKVCache.to_batch([pad]) followed by BatchQSAKVCache.extend() keeps
    the host shadow populated without GPU sync."""
    single = language.QSAKVCache()
    single.update_and_fetch(
        mx.ones((1, 2, 5, 8), dtype=mx.bfloat16),
        mx.ones((1, 2, 5, 8), dtype=mx.bfloat16),
    )
    single.update_indexer(
        mx.ones((1, 5, 8), dtype=mx.bfloat16),
        mx.arange(5, dtype=mx.int32)[None],
    )
    batch_a = single.to_batch([2])
    assert batch_a.kv_cache._left_padding_host == [2]
    assert batch_a.left_padding_list() == [2]

    other = language.QSAKVCache()
    other.update_and_fetch(
        mx.ones((1, 2, 9, 8), dtype=mx.bfloat16),
        mx.ones((1, 2, 9, 8), dtype=mx.bfloat16),
    )
    other.update_indexer(
        mx.ones((1, 9, 8), dtype=mx.bfloat16),
        mx.arange(9, dtype=mx.int32)[None],
    )
    batch_b = other.to_batch([0])
    batch_a.extend(batch_b)
    # batch_a had _idx=7 (5+2), pad=[2]; batch_b has _idx=9, pad=[0] -> [4, 0].
    assert batch_a.kv_cache._left_padding_host == [4, 0]
    assert batch_a.left_padding_list() == [4, 0]
    assert batch_a.left_padding.tolist() == [4, 0]


def test_crossover_boundary_and_padded_below_crossover_behavior():
    """Exact crossover boundary:
    - (width + length) // ratio <= topk returns None without touching left_padding;
    - (width + length) // ratio > topk reads left_padding_list() and still
      returns None when every row's (width - pad + length) // ratio <= topk.
    """
    cfg = _tiny_config(budget=8, ratio=2, attn_layers=1)
    attn = language.Qwen4ExpAttention(cfg)
    mx.eval(attn.parameters())

    # width=6, length=2 -> (6+2)//2 = 4 <= topk=4 -> below crossover.
    c = language.BatchQSAKVCache([0, 0])
    c.update_and_fetch(
        mx.ones((2, 2, 6, 8), dtype=mx.bfloat16),
        mx.ones((2, 2, 6, 8), dtype=mx.bfloat16),
    )
    c.update_indexer(
        mx.ones((2, 6, 8), dtype=mx.bfloat16),
        mx.broadcast_to(mx.arange(6, dtype=mx.int32)[None], (2, 6)),
    )
    canary = _make_canary(7.0)
    c.kv_cache.left_padding = mx.depends(c.kv_cache.left_padding, canary)
    x2 = mx.zeros((2, 2, cfg.hidden_size), dtype=mx.bfloat16)
    mx.synchronize()
    m0 = mx.get_active_memory()
    assert attn._gathered_batch_paddings(x2, "causal", c, None, None, True) is None
    assert mx.get_active_memory() - m0 < canary.nbytes // 2

    # width=8, length=2 -> (8+2)//2 = 5 > topk=4, but pads=[2, 2] ->
    # (8-2+2)//2 = 4 <= topk=4 -> must return None!
    c2 = language.BatchQSAKVCache([2, 2])
    c2.update_and_fetch(
        mx.ones((2, 2, 8, 8), dtype=mx.bfloat16),
        mx.ones((2, 2, 8, 8), dtype=mx.bfloat16),
    )
    c2.update_indexer(
        mx.ones((2, 8, 8), dtype=mx.bfloat16),
        mx.broadcast_to(mx.arange(8, dtype=mx.int32)[None], (2, 8)),
    )
    assert attn._gathered_batch_paddings(x2, "causal", c2, None, None, True) is None

    # pads=[0, 2] -> row 0 has (8-0+2)//2 = 5 > 4 -> returns [0, 2]!
    c3 = language.BatchQSAKVCache([0, 2])
    c3.update_and_fetch(
        mx.ones((2, 2, 8, 8), dtype=mx.bfloat16),
        mx.ones((2, 2, 8, 8), dtype=mx.bfloat16),
    )
    c3.update_indexer(
        mx.ones((2, 8, 8), dtype=mx.bfloat16),
        mx.broadcast_to(mx.arange(8, dtype=mx.int32)[None], (2, 8)),
    )
    assert attn._gathered_batch_paddings(x2, "causal", c3, None, None, True) == [0, 2]


def test_concurrent_ragged_verify_threads_keep_independent_shadows():
    """Rule 14: per-cache host shadows are isolated across engine threads."""
    model_a, base_caches = _build_batch_model(
        budget=8, ratio=2, attn_layers=2, prefixes=(14, 11), seed=41
    )
    model_b, _ = _build_batch_model(
        budget=8, ratio=2, attn_layers=2, prefixes=(14, 11), seed=41
    )
    inp1 = mx.array([[11, 12], [13, 14]], dtype=mx.int32)
    inp2 = mx.array([[15, 16], [17, 18]], dtype=mx.int32)

    def run_worker(mdl, accepted):
        _reset_rope(mdl)
        caches = copy.deepcopy(base_caches)
        out1 = mdl(inp1, cache=caches, return_hidden=True)
        mx.eval(out1.logits)
        mdl.rollback_speculative_cache(caches, out1.gdn_states, accepted, 2)
        out2 = mdl(inp2, cache=caches, return_hidden=True)
        mx.eval(out2.logits)
        pads = [c.left_padding_list() for c in _attn_caches(caches)]
        return out2.logits, pads

    expected_a, pads_a = run_worker(model_a, [1, 0])
    expected_b, pads_b = run_worker(model_b, [0, 1])

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(run_worker, model_a, [1, 0])
        fut_b = pool.submit(run_worker, model_b, [0, 1])
        got_a, got_pads_a = fut_a.result()
        got_b, got_pads_b = fut_b.result()

    assert got_pads_a == pads_a == [[0, 4], [0, 4]]
    assert got_pads_b == pads_b == [[1, 3], [1, 3]]
    _assert_bitwise(got_a, expected_a)
    _assert_bitwise(got_b, expected_b)
