# SPDX-License-Identifier: Apache-2.0
"""Metal HC sinkhorn / collapse / expand parity tests."""
from __future__ import annotations

import mlx.core as mx
import pytest


@pytest.fixture(scope="module")
def patched():
    from omlx.patches.deepseek_v41 import apply_deepseek_v41_patch

    apply_deepseek_v41_patch()


def test_sinkhorn_only_matches_python(patched):
    from omlx.patches.deepseek_v4.hyper_connection import (
        _hc_sinkhorn_only,
        _hc_split_sinkhorn_ops,
    )

    B, L, MIX = 2, 3, 24  # (2+4)*4
    mixes = mx.random.normal((B, L, MIX)).astype(mx.float32)
    scale = mx.array([1.0, 1.0, 1.0], dtype=mx.float32)
    base = mx.zeros((MIX,), dtype=mx.float32)
    ref = _hc_split_sinkhorn_ops(mixes, scale, base, 4, 20, 1e-6)
    fast = _hc_sinkhorn_only(mixes, scale, base, 4, 20, 1e-6, (B, L))
    mx.eval(*ref, *fast)
    for a, b in zip(ref, fast):
        assert float((a - b).abs().max()) < 1e-5


def test_collapse_metal_matches_python(patched):
    from omlx.patches.deepseek_v41.deferred_hc import hc_collapse

    B, L, H, D = 1, 8, 4, 64
    x = mx.random.normal((B, L, H, D)).astype(mx.bfloat16)
    pre = mx.softmax(mx.random.normal((B, L, H)).astype(mx.float32), axis=-1)
    ref = (pre[..., None] * x.astype(mx.float32)).sum(axis=2).astype(x.dtype)
    out = hc_collapse(x, pre)
    mx.eval(ref, out)
    assert float((ref.astype(mx.float32) - out.astype(mx.float32)).abs().max()) < 1e-2


def test_expand_metal_prefill_matches(patched):
    from omlx.patches.deepseek_v4.hyper_connection import hc_expand, _hc_expand_op

    B, L, H, D = 1, 64, 4, 64  # hits metal gate (>=64 tokens)
    x = mx.random.normal((B, L, D)).astype(mx.bfloat16)
    residual = mx.random.normal((B, L, H, D)).astype(mx.bfloat16)
    post = mx.random.normal((B, L, H)).astype(mx.float32)
    comb = mx.softmax(mx.random.normal((B, L, H, H)).astype(mx.float32), axis=-1)
    ref = _hc_expand_op(x, residual, post, comb)
    out = hc_expand(x, residual, post, comb)
    mx.eval(ref, out)
    assert float((ref.astype(mx.float32) - out.astype(mx.float32)).abs().max()) < 5e-2
