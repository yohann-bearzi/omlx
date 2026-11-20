# SPDX-License-Identifier: Apache-2.0
"""Dense prefill over a format's tile face, with one run of expert zero."""

from __future__ import annotations

import logging

import mlx.core as mx

from ..patches import m5_gather_qmm_nax as nax
from . import kernels

logger = logging.getLogger(__name__)


def one_run(kernel, config, x, operands):
    """Apply a dense NAX tile plan to one matrix, preserving x's batch axes."""
    n, k = operands[0].shape[0], x.shape[-1]
    rows = x.size // k
    bm = config.bm
    params = mx.array([n, rows], dtype=mx.uint32)
    y = kernel(
        inputs=[x.reshape(rows, 1, k), *operands, params],
        template=[
            ("T", x.dtype),
            ("K", k),
            ("SCHED", int(config.sched)),
            ("ALIGN_N", n % nax._BN == 0),
            ("BM", bm),
            ("BK", config.bk),
            ("GX", config.gx),
            ("PAD", config.pad),
        ],
        **nax._grid(n, (rows + bm - 1) // bm, config),
        output_shapes=[(*x.shape[:-1], n)],
        output_dtypes=[x.dtype],
    )
    return y[0]


def project(tile, x):
    """Prove the selected tile and replay this thread's dense graph."""
    if tile is None or tile.proof.failed or not tile.face.exact_for(x.dtype):
        return None
    k, n = x.shape[-1], tile.operands[0].shape[0]
    signature = (x.shape, x.dtype, n)
    graph = tile.proof.graphs.get(signature)
    if graph is not None:
        return graph(x, *tile.operands)
    config = nax.face_plan(x.size // k, 1, k, n)
    metal = (x.dtype, k, config, n % 64 == 0)
    kernel = tile.kernel

    def graph(x, *operands):
        return one_run(kernel, config, x, operands)

    def stand_in():
        operands = ((a.shape, a.dtype) for a in tile.operands)
        return kernels.zeros(((16, k), x.dtype), *operands), graph

    if not tile.proof.prove(metal, stand_in, logger, *tile.said):
        return None
    compiled = mx.compile(graph)
    result = compiled(x, *tile.operands)
    tile.proof.keep(signature, compiled)
    return result
