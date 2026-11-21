# SPDX-License-Identifier: Apache-2.0
"""A safetensors file's tensors as stored: the bytes, without MLX.

Each tensor's raw byte range comes from the file's header (dtype, shape,
data_offsets); the digest is over those stored bytes, so it states what a
checkpoint holds, not how a loader happens to lay its arrays out.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

_DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U64": 8,
    "U32": 4,
    "U16": 2,
    "U8": 1,
    "BOOL": 1,
}


def tensor_bytes(path: Path) -> dict[str, dict]:
    """{name: {dtype, shape, sha256}} over each tensor's stored bytes."""
    file_size = path.stat().st_size
    with open(path, "rb") as stream:
        raw_len = stream.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: truncated safetensors header length")
        (length,) = struct.unpack("<Q", raw_len)
        start = 8 + length
        if start > file_size:
            raise ValueError(f"{path}: safetensors header extends past EOF")
        header = json.loads(stream.read(length))
        out = {}
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            dtype = entry["dtype"]
            if dtype not in _DTYPE_BYTES:
                raise ValueError(f"{path}:{name}: unknown safetensors dtype {dtype!r}")
            shape = list(entry["shape"])
            lo, hi = entry["data_offsets"]
            expected = math.prod(shape) * _DTYPE_BYTES[dtype]
            if not (0 <= lo <= hi and start + hi <= file_size):
                raise ValueError(
                    f"{path}:{name}: data_offsets {(lo, hi)} out of bounds"
                )
            if hi - lo != expected:
                raise ValueError(
                    f"{path}:{name}: byte length {hi - lo} != {expected} "
                    f"for {dtype} {shape}"
                )
            stream.seek(start + lo)
            data = stream.read(hi - lo)
            out[name] = {
                "dtype": dtype,
                "shape": shape,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
    return out
