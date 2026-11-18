# SPDX-License-Identifier: Apache-2.0
"""Checked entry points to the frozen sg8 Python and Metal in _sg8."""

import mlx.core as mx

from . import _sg8 as legacy


def _matrix(x, weight, scales, biases, bits, group, *, grouped=False):
    if x.ndim != 2 or x.dtype not in (mx.bfloat16, mx.float16):
        raise ValueError("sg8 requires a matrix of bf16 or fp16 activations")
    rows, k = x.shape
    if weight.ndim != 2 or weight.dtype != mx.uint32:
        raise ValueError("sg8 requires a matrix of packed uint32 words")
    n = weight.shape[0]
    widths = (4, 5) if grouped else (2, 3, 4, 5, 6)
    nsg, ct = (legacy._SG8_GROUP_NSG, 1) if grouped else legacy._sg8_geometry(k, n)
    if (
        not 1 <= rows <= 8
        or min(k, n) <= 0
        or bits not in widths
        or group not in (32, 64, 128)
        or k % (group * nsg)
        or n % (8 * ct)
    ):
        raise ValueError(f"invalid sg8 geometry {(rows, k, n, bits, group)}")
    if weight.shape[1] != k * bits // 32:
        raise ValueError("sg8 weights must contain K codes per row")
    dtypes = (x.dtype,) if grouped else (mx.bfloat16, mx.float16, mx.float32)
    for param in (scales, biases):
        if param.shape != (n, k // group) or param.dtype not in dtypes:
            raise ValueError("invalid sg8 scale or bias tensor")


def _sums(x, group, sums):
    if (
        sums is not None
        and group == legacy._SUM_GROUP
        and (
            sums.shape != (x.shape[0], x.shape[1] // group) or sums.dtype != mx.float32
        )
    ):
        raise ValueError("sg8 needs one float32 sum per row and group")


def vk_qmm_sg8(x2, w_q, scales, biases, *, group_size, bits=4, sums=None):
    _matrix(x2, w_q, scales, biases, bits, group_size)
    _sums(x2, group_size, sums)
    return legacy.vk_qmm_sg8(
        x2, w_q, scales, biases, group_size=group_size, bits=bits, sums=sums
    )


def vk_swiglu_sg8(x2, gate, up, sums=None):
    if (gate.weight.shape, gate.bits, gate.group_size) != (
        up.weight.shape,
        up.bits,
        up.group_size,
    ):
        raise ValueError("sg8 gate and up must have matching dimensions and packing")
    for layer in (gate, up):
        _matrix(
            x2, layer.weight, layer.scales, layer.biases, layer.bits, layer.group_size
        )
    _sums(x2, gate.group_size, sums)
    return legacy.vk_swiglu_sg8(x2, gate, up, sums)


def vk_group_sg8(x2, linears, sums=None):
    if len(linears) < 2:
        raise ValueError("grouped sg8 needs at least two projections")
    group = linears[0].group_size
    for layer in linears:
        if layer.group_size != group:
            raise ValueError(
                "grouped sg8 needs the same group size for every projection"
            )
        _matrix(
            x2,
            layer.weight,
            layer.scales,
            layer.biases,
            layer.bits,
            group,
            grouped=True,
        )
    _sums(x2, group, sums)
    return legacy.vk_group_sg8(x2, linears, sums)


def group_sums(x):
    if not x.ndim or not x.size or x.shape[-1] % legacy._SUM_GROUP:
        raise ValueError("group_sums needs nonempty rows of whole 64-input groups")
    return legacy.group_sums(x)
