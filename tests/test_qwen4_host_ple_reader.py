# SPDX-License-Identifier: Apache-2.0
"""The prefill helper's PLE rows: this Mac decodes them with the table's reader."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from omlx.formats import testing
from omlx.formats.linear import FormatEmbedding
from omlx.prefill_helper import wire
from omlx.prefill_helper.adapters import qwen4_exp as adapter


def _layer(rows, dims, seed):
    w = mx.random.normal((rows, dims), key=mx.random.key(seed)).astype(mx.float32)
    packed = mx.quantize(w, group_size=64, bits=4)
    names = ("weight", "scales", "biases")
    with testing.verified("qwen4_exp"):
        return FormatEmbedding("affine", 4, 64, dims, dict(zip(names, packed)))


def _table(layers, rows):
    offsets, total = [0], 0
    for layer in layers:
        total += layer.output_dims
        offsets.append(total)
    return SimpleNamespace(
        fused=None,
        shards=layers,
        shard_offsets=tuple(offsets),
        dims=layers[0].input_dims,
        weight_scale=mx.ones((1,), dtype=mx.bfloat16),
    )


@pytest.mark.parametrize("shards", [1, 2], ids=("one", "two"))
def test_host_ple_reads_the_formats_rows_and_sends_them_decoded(shards):
    dims = 64
    rows = [5, 7][:shards]
    layers = [_layer(count, dims, 30 + index) for index, count in enumerate(rows)]
    table = _table(layers, rows)
    language = SimpleNamespace(
        DiskBackedShardedEmbedding=type("DiskBackedShardedEmbedding", (), {})
    )
    read = adapter._ple_reader(table, language)
    host = adapter._HostPLERows(dims, mx.bfloat16)
    total = sum(rows)
    if shards == 1:
        ids = np.array([0, total - 1, 1, total // 2], dtype=np.int64)
    else:
        ids = np.array([0, rows[0] - 1, rows[0], total - 1], dtype=np.int64)

    arrays = read(ids)
    inputs = {"p0.ids": (ids, "int64"), "p0.ws": wire.to_host(table.weight_scale)}
    for index, value in enumerate(arrays):
        inputs[f"p0.{index}"] = value
    host.put(inputs, 0)
    got = host(mx.array(ids))
    mx.eval(got)

    if shards == 1:
        expected = layers[0]._decoded(mx.array(ids))
    else:
        expected = mx.concatenate(
            [
                layers[0]._decoded(mx.array(ids[:2])),
                layers[1]._decoded(mx.array(ids[2:] - rows[0])),
            ]
        )
    mx.eval(expected)
    assert got.shape == (*ids.shape, dims)
    np.testing.assert_array_equal(
        np.asarray(got.view(mx.uint16)), np.asarray(expected.view(mx.uint16))
    )


def _bitwise(a, b):
    return a.shape == b.shape and np.array_equal(
        np.asarray(a.view(mx.uint16)), np.asarray(b.view(mx.uint16))
    )


def _quantized_shard(rows, dims, seed):
    mx.random.seed(seed)
    return nn.Embedding(rows, dims).to_quantized(group_size=32, bits=4)


def test_host_ple_reads_mlx_quantized_shards():
    dims = 32
    shards = [_quantized_shard(5, dims, 40), _quantized_shard(7, dims, 41)]
    offsets, total = [0], 0
    for shard in shards:
        total += shard.weight.shape[0]
        offsets.append(total)
    table = SimpleNamespace(
        fused=None,
        shards=shards,
        shard_offsets=tuple(offsets),
        dims=dims,
        weight_scale=mx.ones((1,), dtype=mx.bfloat16),
    )
    language = SimpleNamespace(
        DiskBackedShardedEmbedding=type("DiskBackedShardedEmbedding", (), {})
    )
    read = adapter._ple_reader(table, language)
    ids = np.array([0, 4, 5, total - 1], dtype=np.int64)
    host = adapter._HostPLERows(dims, mx.bfloat16)
    arrays = read(ids)
    inputs = {"p0.ids": (ids, "int64"), "p0.ws": wire.to_host(table.weight_scale)}
    for index, value in enumerate(arrays):
        inputs[f"p0.{index}"] = value
    host.put(inputs, 0)
    got = host(mx.array(ids))
    expected = mx.concatenate(
        [shards[0](mx.array(ids[:2])), shards[1](mx.array(ids[2:] - offsets[1]))]
    )
    mx.eval(got, expected)
    assert _bitwise(got, expected)


def test_host_ple_format_reader_handles_empty_rows():
    dims = 64
    layer = _layer(5, dims, 30)
    table = _table([layer], [5])
    language = SimpleNamespace(
        DiskBackedShardedEmbedding=type("DiskBackedShardedEmbedding", (), {})
    )
    read = adapter._ple_reader(table, language)
    arrays = read(np.array([], dtype=np.int64))
    assert len(arrays) == 1 and arrays[0][0].shape == (0, dims)


def test_host_ple_decode_runs_on_the_serving_gpu_stream():
    dims = 64
    layer = _layer(5, dims, 30)
    table = _table([layer], [5])
    language = SimpleNamespace(
        DiskBackedShardedEmbedding=type("DiskBackedShardedEmbedding", (), {})
    )
    read = adapter._ple_reader(table, language)
    ids = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    with mx.stream(mx.cpu):
        arrays = read(ids)
    got = wire.from_host(*arrays[0])
    expected = layer._decoded(mx.array(ids))
    mx.eval(expected)
    assert _bitwise(got, expected)


def test_disk_ple_reader_handles_empty_rows():
    class DiskTable:
        dims = 32
        weight_scale = mx.ones((1,), dtype=mx.float16)

        def _plan(self, rows):
            if rows.size == 0:
                raise IndexError("no rows")
            return None

    read = adapter._ple_reader(
        DiskTable(), SimpleNamespace(DiskBackedShardedEmbedding=DiskTable)
    )
    arrays = read(np.array([], dtype=np.int64))
    assert len(arrays) == 1
    assert arrays[0][0].shape == (0, 32) and arrays[0][1] == "float16"


def test_disk_ple_reader_casts_rows_to_the_weight_scale_dtype():
    class DiskTable:
        dims = 32
        weight_scale = mx.ones((1,), dtype=mx.float16)

        def _plan(self, rows):
            return (
                None,
                None,
                None,
                None,
                [0, 1, 2],
                {0: "U32", 1: "BF16", 2: "BF16"},
                4,
                32,
            )

        def _assemble(self, rows, plan):
            return {
                0: np.zeros((len(rows), 4), np.uint32),
                1: np.full((len(rows), 1), 0x3F80, np.uint16),
                2: np.zeros((len(rows), 1), np.uint16),
            }

    read = adapter._ple_reader(
        DiskTable(), SimpleNamespace(DiskBackedShardedEmbedding=DiskTable)
    )
    ids = np.array([0, 1], dtype=np.int64)
    arrays = read(ids)
    host = adapter._HostPLERows(DiskTable.dims, mx.float16)
    inputs = {"p0.ids": (ids, "int64"), "p0.ws": wire.to_host(DiskTable.weight_scale)}
    for index, value in enumerate(arrays):
        inputs[f"p0.{index}"] = value
    host.put(inputs, 0)
    got = host(mx.array(ids))
    mx.eval(got)
    assert got.dtype == mx.float16
