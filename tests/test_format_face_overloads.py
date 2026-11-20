# SPDX-License-Identifier: Apache-2.0
"""Read the documented overload even when another one compiles and returns zero."""

import mlx.core as mx
import pytest

from omlx.formats import kernels, testing
from tests import test_formats_kernels as dot
from tests import test_formats_tile as tile

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


def _assert_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert mx.all(expected != 0).item()
    assert mx.array_equal(actual.view(mx.uint8), expected.view(mx.uint8)).item()


def _dot_outputs(face, path):
    n, k = 8, 384  # One full block and a partial block.
    weights = (129 + mx.arange(n * k).reshape(n, k) % 7).astype(mx.uint8)
    scales = mx.arange(1, n + 1).astype(mx.float32)
    x = (1 + mx.arange(k) % 4).astype(mx.float32)
    w = (weights.astype(mx.float32) - 128) * scales[:, None]
    if path == "parts":
        stepped, direct = testing.lane_parts(face, (weights, scales), x, n)
        count = k // face.vpt
        expected = mx.sum(
            ((weights.astype(mx.float32) - 128) * x).reshape(n, count, face.vpt), axis=2
        )
        for output in (stepped, direct):
            _assert_bits(output[:, :count], expected)
            assert mx.all(output[:, count:] == 0).item()
        return direct[:, :count], expected
    if path in ("dot", "staged"):
        actual = testing.dot_rows(
            face, (weights, scales), x, n, staged=path == "staged"
        )
        return actual, mx.sum(w * x, axis=1)
    xs = mx.stack([x, x * 2])
    actual = testing.shared_rows(
        face, (weights, scales), xs, n, rps=2, unrolled=path == "unrolled"
    )
    return actual, mx.sum(xs[:, None, :] * w, axis=2)


def _dot_reader(overloaded):
    source = dot._TOY
    if overloaded:
        anchor = "    float finish("
        assert source.count(anchor) == 1
        source = source.replace(
            anchor,
            """
    static float load(device float*, thread float* xt) {
      for (int i = 0; i < VPT; ++i) xt[i] = 0;
      return 0;
    }
    static float load(threadgroup float*, thread float* xt) {
      for (int i = 0; i < VPT; ++i) xt[i] = 0;
      return 0;
    }
    static float madd(float, Row, thread float*, float) { return 0; }
    template <typename A>
    static float apply(float, thread const Row&, thread float*, A) { return 0; }
""" + anchor,
        )
    return dot._toy(f"xt_{int(overloaded)}", source)


@pytest.mark.parametrize("path", ["dot", "staged", "shared", "unrolled", "parts"])
@pytest.mark.parametrize("overloaded", [False, True], ids=["const", "mutable"])
def test_dot_reads_const_activation_scratch(path, overloaded):
    actual, expected = _dot_outputs(_dot_reader(overloaded), path)
    _assert_bits(actual, expected)


@pytest.mark.parametrize("overload", ["const", "params", "codes", "short"])
@pytest.mark.parametrize("quad", [False, True], ids=["scalar", "fetch4"])
def test_tile_reads_const_weight_with_int_index(quad, overload):
    n, k = 8, 72  # Two complete quads and one scalar run.
    weights = (129 + mx.arange(n * k).reshape(n, k) % 17).astype(mx.uint8)
    group = mx.arange(1, n * 3 + 1).reshape(n, 3).astype(mx.float32)
    scale = mx.arange(1, n + 1).astype(mx.float32)
    source = tile._TOY
    signature = "    static float weight(thread const P& p, Codes q, int i) {"
    assert source.count(signature) == 1
    extra = {
        "const": "",
        "params": (
            "    static float weight(thread P&, "
            "thread const Codes&, int) { return 0; }\n"
        ),
        "codes": (
            "    static float weight(thread const P&, "
            "thread Codes&, int) { return 0; }\n"
        ),
        "short": (
            "    static float weight(thread const P&, "
            "thread const Codes&, short) { return 0; }\n"
        ),
    }[overload]
    if quad:
        extra += """    void fetch4(int quad, thread Codes* out) const {
      for (int i = 0; i < 4; ++i) out[i] = fetch(4 * quad + i);
    }
"""
    face = tile._toy(
        f"read_{overload}_{int(quad)}",
        source.replace(
            signature, extra + signature.replace("Codes q", "thread const Codes& q")
        ),
    )
    rows = [7, 0, 3]
    want = tile._toy_decode(weights, group, scale)[rows]
    exact, rounded = testing.tile_rows(
        face, (weights, group, scale), k, rows, mx.bfloat16
    )
    _assert_bits(exact, want)
    _assert_bits(rounded, want.astype(mx.bfloat16))


@pytest.mark.parametrize("shared", [False, True], ids=["dot", "shared"])
@pytest.mark.parametrize("staged", [False, True], ids=["device", "threadgroup"])
def test_skeleton_qualifies_a_callers_mutable_activation_pointer(shared, staged):
    # Callers can own writable activation storage. The face only reads it.
    face = _dot_reader(True)
    matrices = kernels.Matrices()
    operands = matrices.add("m", face, shared=shared)
    storage = (
        "threadgroup float xs[256];"
        "for (int i = lane; i < 256; i += 32) xs[i] = x[i];"
        "threadgroup_barrier(mem_flags::mem_threadgroup);"
        if staged
        else "device float* xs = const_cast<device float*>(x);"
    )
    call = (
        "qmv_shared<float,256,1,1,8,false>(m,0,1,xs,lane,result);"
        if shared
        else "qmv_rows<float,256,1,0>(m,0,m,0,xs,lane,result);"
    )
    kernel = mx.fast.metal_kernel(
        name=f"kit_mutable_storage_{int(shared)}_{int(staged)}",
        input_names=["x", *operands],
        output_names=["out"],
        header=matrices.header,
        source=matrices.declarations
        + "uint lane = thread_index_in_simdgroup; float result[1];"
        + storage
        + call
        + "if (lane == 0) out[0] = result[0];",
    )
    (got,) = kernel(
        # Eight matrix rows keep scales in device memory in the generated kernel.
        inputs=[
            mx.ones((256,)),
            mx.full((8, 256), 130, mx.uint8),
            mx.arange(1, 9, dtype=mx.float32),
        ],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1,)],
        output_dtypes=[mx.float32],
    )
    _assert_bits(got, mx.array([512.0]))
