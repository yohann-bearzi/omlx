# SPDX-License-Identifier: Apache-2.0
"""The qwen4 MTP head under a format load: its quantization paths resolve
and the plan-less affine row-exact pin holds."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()

from mlx_vlm.models.qwen4_exp import Model  # noqa: E402

PREFIXES = (
    "mtp.",
    "model.mtp.",
    "language_model.mtp.",
    "model.language_model.mtp.",
    "language_model.model.mtp.",
)


def test_qwen4_quantization_path_aliases_resolve_every_mtp_prefix():
    model = Model.__new__(Model)
    aliases = model.quantization_path_aliases("mtp.fc_embedding")
    assert set(aliases) == {prefix + "fc_embedding" for prefix in PREFIXES}
    assert len(aliases) == len(PREFIXES)


def test_qwen4_quantization_path_aliases_non_mtp_is_empty():
    model = Model.__new__(Model)
    assert (
        model.quantization_path_aliases("language_model.model.layers.0.mlp.gate")
        == ()
    )
    assert model.quantization_path_aliases("mtpx.fc_embedding") == ()
    assert model.quantization_path_aliases("model.mtp.fc_embedding") == ()
    assert model.quantization_path_aliases("foo.mtp.bar") == ()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("rows", [1, 4])
def test_planless_stock_affine_keeps_mlx_matmul(monkeypatch, dtype, rows):
    """An affine layer whose face declines the matrix (a 128-deep K, or a
    matrix under one group) keeps MLX's own matmul, batched at every row
    count: Rule 33 retired row-exact arming, so an armed flag no longer
    splits the call into rows, and the decoded float matmul is never used."""
    import mlx.core as mx
    import mlx.nn as nn

    from omlx.formats import linear, testing
    from omlx.formats.linear import FormatLinear
    from omlx.patches import qwen35_verify_qmm as verify
    from tests.test_formats_linear import _tensors

    k, n, bits, group_size = 128, 256, 8, 128
    mx.random.seed(3)
    dense = nn.Linear(k, n, bias=False)
    dense.set_dtype(dtype)
    quantized = dense.to_quantized(group_size=group_size, bits=bits)
    verify.set_verify_qmm_armed(True, row_exact=True)
    try:
        with testing.verified("qwen4_exp"):
            layer = FormatLinear(
                "affine",
                bits,
                group_size,
                k,
                _tensors(quantized),
                bias=quantized.get("bias"),
            )
        assert linear._plan(layer) is None
        x = mx.random.normal((rows, k)).astype(dtype)
        batched = mx.quantized_matmul(
            x,
            quantized["weight"],
            quantized["scales"],
            quantized.get("biases"),
            transpose=True,
            group_size=group_size,
            bits=bits,
        )
        got = layer(x)
        mx.eval(got, batched)
        assert mx.array_equal(got.view(mx.uint16), batched.view(mx.uint16)).item()
        monkeypatch.setattr(
            FormatLinear, "_decoded", lambda self: pytest.fail("decoded the matrix")
        )
        again = layer(x)
        mx.eval(again)
        assert mx.array_equal(again.view(mx.uint16), batched.view(mx.uint16)).item()
    finally:
        verify.set_verify_qmm_armed(False)


def test_qwen4_mtp_quantization_aliases_resolve_through_mlx_vlm():
    """Pin the load-time caller of ``quantization_path_aliases``: the
    release pins mlx-vlm ``138d761`` (0.7.11), whose ``utils.py:292-310``
    (``_quantization_path_aliases`` -> ``Model.quantization_path_aliases``)
    feeds ``_quantization_for_module_path`` at ``nn.quantize`` (utils.py:651).
    A checkpoint key under HF's or the sanitizer's MTP spelling must resolve
    at the runtime ``mtp.`` path; without the model's aliases it does not."""
    import mlx_vlm.utils as vlm_utils

    model = Model.__new__(Model)
    entry = {"bits": 8, "group_size": 64, "mode": "affine"}
    for spelling in (
        "model.language_model.mtp.fc_embedding",
        "model.mtp.fc_embedding",
        "language_model.model.mtp.fc_embedding",
    ):
        found = vlm_utils._quantization_for_module_path(
            {spelling: entry}, "mtp.fc_embedding", model=model
        )
        assert found is entry, spelling
    assert (
        vlm_utils._quantization_for_module_path(
            {"model.mtp.fc_embedding": entry}, "mtp.fc_embedding"
        )
        is None
    )
