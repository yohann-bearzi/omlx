"""Correctness tests for the oQ mixed-bit QxA8 prefill kernels.

Kernel correctness and model accuracy are separate concerns, and the kernel
tests come first.

  1. decoded Q4/Q5/Q8 codes are bit-exact against an independent bit unpacker,
     which is in turn tied to MLX's own dequantize semantics
  2. Stage A's Qa is bit-exact and Ra is exactly the sum of the codes
  3. the INT32 dot and the FP32 affine accumulation match an independent
     implementation

Everything that needs the native extension is skipped when it is absent, so
this file is meaningful on a default install: the reference implementations and
the classification logic are exercised regardless.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

GROUP_SIZE = 64


def _kernels():
    """The native module, or None when this build/host cannot run it."""
    try:
        from omlx.custom_kernels.quant_matmul import fast
    except Exception:
        return None
    if not fast.oq_a8_available():
        return None
    return fast


requires_kernels = pytest.mark.skipif(
    _kernels() is None,
    reason="oQ A8 NAX kernels are unavailable (no native build or no tensor units)",
)


# --------------------------------------------------------------------------
# Independent references
# --------------------------------------------------------------------------


def unpack_codes(packed: np.ndarray, bits: int, K: int) -> np.ndarray:
    """Unpack MLX's affine uint32 code stream, independently of the kernel.

    Codes run little-endian along the bit stream of each row, so a row of K
    codes occupies exactly K * bits / 32 words -- which is why GS64 lands on a
    word boundary for both Q4 (8 words) and Q5 (10 words).
    """
    n_rows = packed.shape[0]
    stream = np.unpackbits(
        packed.view(np.uint8).reshape(n_rows, -1), axis=1, bitorder="little"
    )
    starts = np.arange(K) * bits
    codes = np.zeros((n_rows, K), dtype=np.int64)
    for b in range(bits):
        codes |= stream[:, starts + b].astype(np.int64) << b
    return codes


def quantize_activations(x: np.ndarray, act_mode: int):
    """Reference Stage A.

    rint() is roundTiesToEven in both Metal and numpy, so the codes compare
    exactly rather than within a tolerance.
    """
    x = x.astype(np.float32)
    rows, K = x.shape
    groups = K // GROUP_SIZE

    if act_mode == 0:
        amax = np.abs(x).max(axis=1)
        scale = np.where(amax > 0, amax / 127.0, 0.0).astype(np.float32)
        inv = np.where(amax > 0, 127.0 / amax, 0.0).astype(np.float32)
        qa = np.clip(np.rint(x * inv[:, None]), -127, 127).astype(np.int8)
    else:
        blocks = x.reshape(rows, groups, GROUP_SIZE)
        amax = np.abs(blocks).max(axis=2)
        scale = np.where(amax > 0, amax / 127.0, 0.0).astype(np.float32)
        inv = np.where(amax > 0, 127.0 / amax, 0.0).astype(np.float32)
        qa = np.clip(np.rint(blocks * inv[:, :, None]), -127, 127)
        qa = qa.astype(np.int8).reshape(rows, K)

    ra = qa.reshape(rows, groups, GROUP_SIZE).sum(axis=2, dtype=np.int64)
    # |Ra| <= 64 * 127 = 8128 always fits INT16.
    assert np.abs(ra).max() <= GROUP_SIZE * 127
    return qa, scale, ra.astype(np.int16)


def affine_reference(qa, sa, ra, codes, sw, bw, act_mode) -> np.ndarray:
    """Reference for the group-accumulated affine GEMM.

    Deliberately written the long way -- integer dot per group, then scale and
    bias -- rather than by dequantizing the weights, so it is an independent
    check on the accumulator hierarchy and not a restatement of it.
    """
    rows, K = qa.shape
    groups = K // GROUP_SIZE
    n_out = codes.shape[0]

    qa_g = qa.reshape(rows, groups, GROUP_SIZE).astype(np.int64)
    qw_g = codes.reshape(n_out, groups, GROUP_SIZE).astype(np.int64)

    out = np.zeros((rows, n_out), dtype=np.float64)
    for g in range(groups):
        # D[m, n] = sum_k Qa * Qw, exact in integer arithmetic.
        d = qa_g[:, g, :] @ qw_g[:, g, :].T
        contribution = sw[None, :, g] * d + bw[None, :, g] * ra[:, g, None]
        scale = sa[:, None] if act_mode == 0 else sa[:, g, None]
        out += scale * contribution
    return out


def make_quantized(n_out: int, K: int, bits: int, dtype=mx.float16, seed: int = 0):
    rng = np.random.default_rng(seed)
    w = mx.array(rng.standard_normal((n_out, K)).astype(np.float32) * 0.05, dtype=dtype)
    packed, scales, biases = mx.quantize(
        w, group_size=GROUP_SIZE, bits=bits, mode="affine"
    )
    mx.eval(packed, scales, biases)
    return packed, scales, biases


# --------------------------------------------------------------------------
# 1. Decoder
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_unpack_matches_mlx_dequantize(bits):
    """Tie the reference unpacker to MLX's affine semantics.

    If this holds, the decoded codes really are the codes MLX would have used,
    so the kernel comparisons below are anchored to the checkpoint format and
    not just to the test's own idea of it.
    """
    K = 256
    packed, scales, biases = make_quantized(64, K, bits)
    codes = unpack_codes(np.array(packed), bits, K)

    assert codes.min() >= 0
    assert codes.max() <= (1 << bits) - 1

    s = np.array(scales.astype(mx.float32))
    b = np.array(biases.astype(mx.float32))
    expected = np.repeat(s, GROUP_SIZE, axis=1) * codes + np.repeat(
        b, GROUP_SIZE, axis=1
    )
    got = np.array(
        mx.dequantize(
            packed, scales, biases, group_size=GROUP_SIZE, bits=bits, mode="affine"
        ).astype(mx.float32)
    )
    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("bits", [4, 5])
def test_group_words(bits):
    """Q4 uses 8 uint32 per group and Q5 uses 10 -- both word-aligned."""
    K = 512
    packed, _, _ = make_quantized(64, K, bits)
    words_per_group = (GROUP_SIZE * bits) // 32
    assert packed.shape[1] == (K // GROUP_SIZE) * words_per_group


@requires_kernels
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_kernel_decode_is_bit_exact(bits):
    """Decoded codes match the reference unpacker (Q8 centered)."""
    fast = _kernels()
    K = 512
    packed, _, _ = make_quantized(96, K, bits, seed=bits)

    decoded = fast.oq_a8_decode_weights(packed, bits, K // GROUP_SIZE)
    mx.eval(decoded)

    expected = unpack_codes(np.array(packed), bits, K) - (128 if bits == 8 else 0)
    np.testing.assert_array_equal(np.array(decoded).astype(np.int64), expected)


@requires_kernels
def test_q8_decode_edge_codes():
    """0, 127, 128 and 255 straddle the centering sign flip."""
    fast = _kernels()
    codes = np.tile(
        np.array([0, 127, 128, 255, 1, 126, 129, 254], dtype=np.uint8), (4, 8)
    )
    packed = mx.array(codes.view("<u4").copy())
    decoded = fast.oq_a8_decode_weights(packed, 8, 1)
    mx.eval(decoded)
    np.testing.assert_array_equal(
        np.array(decoded).astype(np.int64), codes.astype(np.int64) - 128
    )


# --------------------------------------------------------------------------
# 2. Stage A
# --------------------------------------------------------------------------


@requires_kernels
@pytest.mark.parametrize("act_mode", [0, 1])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_stage_a_matches_reference(act_mode, dtype):
    fast = _kernels()
    rows, K = 96, 256
    rng = np.random.default_rng(7)
    x_np = (rng.standard_normal((rows, K)) * 0.7).astype(np.float32)
    x = mx.array(x_np, dtype=dtype)

    qa, sa, ra = fast.oq_a8_quantize(x, act_mode)
    mx.eval(qa, sa, ra)

    # Quantize from the same rounded input the kernel saw, so the comparison
    # isolates the kernel rather than the host-side cast.
    x_ref = np.array(x.astype(mx.float32))
    qa_ref, sa_ref, ra_ref = quantize_activations(x_ref, act_mode)

    np.testing.assert_array_equal(np.array(qa), qa_ref)
    np.testing.assert_array_equal(np.array(ra), ra_ref)
    np.testing.assert_allclose(
        np.array(sa).reshape(sa_ref.shape), sa_ref, rtol=1e-6, atol=1e-8
    )


@requires_kernels
@pytest.mark.parametrize("act_mode", [0, 1])
def test_stage_a_group_sum_is_exactly_the_code_sum(act_mode):
    """Ra must come from the rounded codes, not from x."""
    fast = _kernels()
    rows, K = 64, 320
    rng = np.random.default_rng(11)
    x = mx.array((rng.standard_normal((rows, K)) * 2.0).astype(np.float32), mx.float16)

    qa, _, ra = fast.oq_a8_quantize(x, act_mode)
    mx.eval(qa, ra)

    codes = np.array(qa).astype(np.int64).reshape(rows, K // GROUP_SIZE, GROUP_SIZE)
    np.testing.assert_array_equal(np.array(ra).astype(np.int64), codes.sum(axis=2))


@requires_kernels
def test_stage_a_zero_row_is_not_nan():
    """An all-zero row has no representable scale; it must not produce NaN."""
    fast = _kernels()
    x = mx.zeros((64, 128), dtype=mx.float16)
    qa, sa, ra = fast.oq_a8_quantize(x, 0)
    mx.eval(qa, sa, ra)

    assert not np.isnan(np.array(sa)).any()
    assert not np.array(qa).any()
    assert not np.array(ra).any()


# --------------------------------------------------------------------------
# 3. GEMM
# --------------------------------------------------------------------------


@requires_kernels
@pytest.mark.parametrize("bits", [4, 5, 6])
@pytest.mark.parametrize("act_mode", [0, 1])
def test_qmm_matches_affine_reference(bits, act_mode):
    """FP32 affine accumulation against an independent implementation.

    The integer dot inside is exact, so any disagreement beyond FP32 rounding
    means the group accumulator or the correction is wrong.
    """
    fast = _kernels()
    M, K, N = 128, 256, 128
    packed, scales, biases = make_quantized(N, K, bits, seed=bits + 3)

    rng = np.random.default_rng(23)
    x = mx.array((rng.standard_normal((M, K)) * 0.5).astype(np.float32), mx.float16)

    qa, sa, ra = fast.oq_a8_quantize(x, act_mode)
    qa8, sa8, ra8 = fast.oq_a8_stage_a_v8(x, act_mode)
    got = fast.oq_a8_qmm_t(
        qa8,
        sa8,
        ra8,
        packed,
        mx.contiguous(scales.T),
        mx.contiguous(biases.T),
        bits,
        act_mode,
        800,
    )
    mx.eval(qa, sa, ra, got)

    expected = affine_reference(
        np.array(qa).astype(np.int64),
        np.array(sa).reshape(M, -1).squeeze() if act_mode == 0 else np.array(sa),
        np.array(ra).astype(np.int64),
        unpack_codes(np.array(packed), bits, K),
        np.array(scales.astype(mx.float32)),
        np.array(biases.astype(mx.float32)),
        act_mode,
    )

    got_np = np.array(got.astype(mx.float32))
    scale = max(np.abs(expected).max(), 1e-6)
    np.testing.assert_allclose(got_np, expected, rtol=2e-2, atol=3e-3 * scale)


@requires_kernels
@pytest.mark.parametrize("bits", [4, 5, 8])
def test_qmm_tracks_the_unquantized_projection(bits):
    """Accuracy sanity: A8 changes numerics but must not change the answer.

    This is not the end-to-end accuracy gate -- that one runs on the model,
    not on a single GEMM -- but a kernel that is merely self-consistent and
    wrong would pass the reference test above and fail here.
    """
    fast = _kernels()
    M, K, N = 128, 512, 128
    packed, scales, biases = make_quantized(N, K, bits, seed=bits + 9)

    rng = np.random.default_rng(31)
    x = mx.array((rng.standard_normal((M, K)) * 0.5).astype(np.float32), mx.float16)

    got = fast.oq_a8_linear(x, packed, scales, biases, bits, 0, 800)
    reference = mx.quantized_matmul(
        x,
        packed,
        scales,
        biases,
        transpose=True,
        group_size=GROUP_SIZE,
        bits=bits,
        mode="affine",
    )
    mx.eval(got, reference)

    got_np = np.array(got.astype(mx.float32))
    ref_np = np.array(reference.astype(mx.float32))
    rel = np.abs(got_np - ref_np).max() / max(np.abs(ref_np).max(), 1e-6)
    assert rel < 0.05, f"A8 projection drifted {rel:.4f} from the W{bits}A16 result"


@requires_kernels
def test_qmm_handles_a_partial_row_tile():
    """M is the token count and need not tile; N is checked host-side."""
    fast = _kernels()
    M, K, N = 130, 128, 64  # 130 is not a multiple of any BM
    packed, scales, biases = make_quantized(N, K, 4, seed=13)
    rng = np.random.default_rng(53)
    x = mx.array((rng.standard_normal((M, K)) * 0.5).astype(np.float32), mx.float16)

    got = fast.oq_a8_linear(x, packed, scales, biases, 4, 0, 800)
    mx.eval(got)
    assert got.shape == (M, N)
    assert not np.isnan(np.array(got.astype(mx.float32))).any()


@requires_kernels
def test_qmm_rejects_untiled_output_width():
    fast = _kernels()
    K, N = 128, 96  # 96 is not a multiple of BN=64
    packed, scales, biases = make_quantized(N, K, 4, seed=17)
    x = mx.zeros((64, K), dtype=mx.float16)
    qa, sa, ra = fast.oq_a8_quantize(x, 0)
    with pytest.raises(ValueError):
        fast.oq_a8_linear(x, packed, scales, biases, 4, 0, 800)


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


def _quantized_linear(in_dim, out_dim, bits, group_size=GROUP_SIZE):
    """A QuantizedLinear shaped like a real checkpoint layer.

    nn.QuantizedLinear defaults to float32 scales; checkpoints store them at
    the activation dtype. Without the cast the classifier correctly rejects the
    module and every dispatch assertion below would pass vacuously.
    """
    import mlx.nn as nn

    linear = nn.QuantizedLinear(
        in_dim, out_dim, bias=False, group_size=group_size, bits=bits
    )
    linear.set_dtype(mx.float16)
    return linear


def _opt_in(obj, min_tokens=128):
    """Tag ``obj`` (and its submodules) the way an enabled model is tagged."""
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    config = dispatch.OqA8Config(min_tokens=min_tokens)
    for module in obj.modules() if isinstance(obj, nn.Module) else [obj]:
        setattr(module, dispatch._CONFIG_ATTR, config)
    return obj


@requires_kernels
def test_classification_is_frozen_and_memoized():
    """The forward path must never re-parse quantization config."""
    from omlx.patches import qwen35_oq_a8 as dispatch

    linear = _quantized_linear(256, 128, 4)
    first = dispatch.classify_linear(linear)
    second = dispatch.classify_linear(linear)
    assert first is second
    assert first is not None, "a checkpoint-shaped Q4 GS64 layer must classify"
    assert first.bits == 4
    assert first.group_size == 64
    assert first.kernel == "q4a8_g64"


@requires_kernels
def test_float32_scales_are_not_claimed():
    """Only checkpoint-dtype scales route here; float32 stays on MLX."""
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    linear = nn.QuantizedLinear(256, 128, bias=False, group_size=64, bits=4)
    assert linear.scales.dtype == mx.float32
    assert dispatch.classify_linear(linear) is None


@requires_kernels
@pytest.mark.parametrize("variant", [800, 803, 806])
def test_dispatch_runs_and_tracks_the_original(monkeypatch, variant):
    """oq_a8_linear must actually route to the kernel, not fall back."""
    from omlx.patches import qwen35_oq_a8 as dispatch

    monkeypatch.setattr(dispatch, "_DEFAULT_VARIANT_Q4", variant)
    linear = _opt_in(_quantized_linear(512, 128, 4), min_tokens=64)
    plan = dispatch.classify_linear(linear)
    assert plan is not None and plan.variant == variant

    rng = np.random.default_rng(3)
    x = mx.array((rng.standard_normal((128, 512)) * 0.5).astype(np.float32), mx.float16)
    got = dispatch.oq_a8_linear(linear, x)
    reference = linear(x)
    mx.eval(got, reference)

    assert hasattr(linear, "_omlx_oq_a8_prepared"), "operand transform not cached"

    g = np.array(got.astype(mx.float32))
    r = np.array(reference.astype(mx.float32))
    # A8 changes numerics, so this must differ from W4A16 -- but only a little.
    # An exact match would mean the dispatcher silently fell back.
    rel = np.abs(g - r).max() / max(np.abs(r).max(), 1e-6)
    assert 0.0 < rel < 0.05, f"dispatch did not reach the kernel (rel={rel})"


@pytest.mark.parametrize("bits", [2, 6])
def test_unsupported_bit_widths_are_not_claimed(bits):
    """Only Q4, Q5 and Q8 are production paths here; the rest stay on MLX."""
    from omlx.patches import qwen35_oq_a8 as dispatch

    assert dispatch.classify_linear(_quantized_linear(256, 128, bits)) is None


@requires_kernels
@pytest.mark.parametrize(
    "bits, kernel", [(4, "q4a8_g64"), (5, "q5a8_g64"), (8, "q8a8_g64")]
)
def test_gs64_projections_have_a8_plan(bits, kernel):
    from omlx.patches import qwen35_oq_a8 as dispatch

    plan = dispatch.classify_linear(_quantized_linear(256, 128, bits))
    assert plan is not None
    assert (plan.bits, plan.group_size, plan.kernel) == (bits, 64, kernel)


def test_q6_projections_have_no_a8_plan():
    """Q6 stays on the A16 path, at either supported group size."""
    from omlx.patches import qwen35_oq_a8 as dispatch

    for group_size in (64, 128):
        linear = _quantized_linear(256, 128, 6, group_size=group_size)
        assert dispatch.classify_linear(linear) is None


@requires_kernels
@pytest.mark.parametrize("bits", [4, 5, 8])
def test_output_width_that_does_not_tile_has_no_a8_plan(bits):
    """N must be a multiple of the tile width (64). Qwen3.8's linear-attention
    in_proj_a / in_proj_b are N=48, so those projections stay on the A16 path."""
    from omlx.patches import qwen35_oq_a8 as dispatch

    assert dispatch.classify_linear(_quantized_linear(256, 48, bits)) is None
    assert dispatch.classify_linear(_quantized_linear(256, 64, bits)) is not None


@pytest.mark.parametrize("bits", [4, 8])
def test_group_size_128_is_not_claimed(bits):
    from omlx.patches import qwen35_oq_a8 as dispatch

    linear = _quantized_linear(256, 128, bits, group_size=128)
    assert dispatch.classify_linear(linear) is None


def test_disabled_without_opt_in():
    """Turning this on changes inference numerics, so it must be explicit."""
    from omlx.patches import qwen35_oq_a8 as dispatch

    assert dispatch._config_for(_quantized_linear(256, 128, 4)) is None


# --------------------------------------------------------------------------
# Step-transposed GEMM (variants >= 800)
# --------------------------------------------------------------------------


def _v8_act_from():
    """Fragment slot -> K within an affine group, derived from the schedule.

    Slot ``16c + 4t + j`` of the group carries ``k = 16c + 8*(t>>1) + 2j +
    (t&1)``: the assignment under which a Q4 step is one nibble parity of one
    word of the untouched checkpoint stream.
    """
    a_from = np.empty(GROUP_SIZE, dtype=np.int64)
    for c in range(4):
        for t in range(4):
            for j in range(4):
                a_from[16 * c + 4 * t + j] = 16 * c + 8 * (t >> 1) + 2 * j + (t & 1)
    return a_from


@requires_kernels
@pytest.mark.parametrize("act_mode", [0, 1])
def test_stage_a_v8_carries_the_schedules_k_order(act_mode):
    """Qa is the only operand that moves, and it moves by exactly this map."""
    fast = _kernels()
    M, K = 33, 256
    rng = np.random.default_rng(21)
    x = mx.array((rng.standard_normal((M, K)) * 0.5).astype(np.float32), mx.float16)
    mx.eval(x)

    qa, sa, ra = fast.oq_a8_quantize(x, act_mode)
    qa8, sa8, ra8 = fast.oq_a8_stage_a_v8(x, act_mode)
    mx.eval(qa, sa, ra, qa8, sa8, ra8)

    groups = K // GROUP_SIZE
    plain = np.array(qa).reshape(M, groups, GROUP_SIZE)
    np.testing.assert_array_equal(
        np.array(qa8).reshape(M, groups, GROUP_SIZE), plain[:, :, _v8_act_from()]
    )
    # Group membership is untouched, which is what keeps Ra and the affine
    # scales valid under the reorder.
    np.testing.assert_array_equal(
        np.sort(plain, axis=2),
        np.sort(np.array(qa8).reshape(M, groups, GROUP_SIZE), axis=2),
    )
    np.testing.assert_array_equal(np.array(ra8), np.array(ra).T)
    expected_sa = np.array(sa).T if act_mode else np.array(sa)
    np.testing.assert_array_equal(np.array(sa8), expected_sa)


@requires_kernels
def test_v8_reads_the_checkpoint_weight_stream_unchanged():
    """The whole point: no repacked weight array, so nothing is held twice."""
    from omlx.patches import qwen35_oq_a8 as dispatch

    linear = _quantized_linear(512, 128, 4)
    assert dispatch.classify_linear(linear) is not None
    weight, scales, biases = dispatch._prepared_weights(linear)
    assert weight is linear.weight, "the packed stream must be reused, not copied"
    np.testing.assert_array_equal(np.array(scales), np.array(linear.scales).T)
    np.testing.assert_array_equal(np.array(biases), np.array(linear.biases).T)


@requires_kernels
@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("act_mode", [0, 1])
@pytest.mark.parametrize("M", [1, 17, 128, 513])
def test_v8_matches_w4a16_within_quantization_error(bits, act_mode, M):
    """End to end off the unmodified checkpoint layout."""
    fast = _kernels()
    K, N = 320, 128
    rng = np.random.default_rng(hash((bits, act_mode, M, 8)) % 2**31)
    w = mx.array(rng.standard_normal((N, K)).astype(np.float32) * 0.05, mx.float16)
    packed, scales, biases = mx.quantize(
        w, group_size=GROUP_SIZE, bits=bits, mode="affine"
    )
    x = mx.array((rng.standard_normal((M, K)) * 0.5).astype(np.float32), mx.float16)
    mx.eval(packed, scales, biases, x)

    sc_t = mx.contiguous(scales.T)
    bi_t = mx.contiguous(biases.T)
    qa, sa, ra = fast.oq_a8_stage_a_v8(x, act_mode)
    mx.eval(sc_t, bi_t, qa, sa, ra)

    got = fast.oq_a8_qmm_t(qa, sa, ra, packed, sc_t, bi_t, bits, act_mode, 800)
    ref = mx.quantized_matmul(
        x,
        packed,
        scales,
        biases,
        transpose=True,
        group_size=GROUP_SIZE,
        bits=bits,
        mode="affine",
    )
    mx.eval(got, ref)
    g = np.array(got.astype(mx.float32))
    r = np.array(ref.astype(mx.float32))
    assert np.isfinite(g).all()
    rel = np.abs(g - r).max() / max(np.abs(r).max(), 1e-6)
    assert 0.0 < rel < 0.06, f"rel={rel}"


@requires_kernels
@pytest.mark.parametrize("variant", [800, 801, 802, 803, 804, 805, 806])
@pytest.mark.parametrize("bits", [4, 5, 6])
def test_v8_tiles_agree_with_each_other(variant, bits):
    """Every tile computes the same thing; only the simdgroup grid differs."""
    fast = _kernels()
    M, K, N = 96, 256, 128
    rng = np.random.default_rng(5)
    w = mx.array(rng.standard_normal((N, K)).astype(np.float32) * 0.05, mx.float16)
    packed, scales, biases = mx.quantize(
        w, group_size=GROUP_SIZE, bits=bits, mode="affine"
    )
    x = mx.array((rng.standard_normal((M, K)) * 0.5).astype(np.float32), mx.float16)
    mx.eval(packed, scales, biases, x)
    sc_t = mx.contiguous(scales.T)
    bi_t = mx.contiguous(biases.T)
    qa, sa, ra = fast.oq_a8_stage_a_v8(x, 0)
    mx.eval(sc_t, bi_t, qa, sa, ra)

    base = fast.oq_a8_qmm_t(qa, sa, ra, packed, sc_t, bi_t, bits, 0, 800)
    got = fast.oq_a8_qmm_t(qa, sa, ra, packed, sc_t, bi_t, bits, 0, variant)
    mx.eval(base, got)
    np.testing.assert_allclose(
        np.array(got.astype(mx.float32)),
        np.array(base.astype(mx.float32)),
        rtol=0,
        atol=0,
    )


@requires_kernels
@pytest.mark.parametrize("variant", [800, 801, 802, 803, 804, 805, 806])
@pytest.mark.parametrize("act_mode", [0, 1])
def test_packed_linear_matches_the_row_major_layout(variant, act_mode):
    """A PackedLinear routes to A8 and computes exactly what its source did.

    Two projections share one store, so the second reads from a tile offset.
    """
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch
    from omlx.patches.qwen35_packed_linear import _pack

    fast = _kernels()
    M, K, N = 96, 512, 384
    sources = []
    for _ in range(2):
        linear = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=4)
        linear.set_dtype(mx.bfloat16)
        sources.append(linear)
    packed = _pack(sources)
    rng = np.random.default_rng(9)
    x = mx.array((rng.standard_normal((M, K)) * 0.5).astype(np.float32), mx.bfloat16)
    qa, sa, ra = fast.oq_a8_stage_a_v8(x, act_mode)

    for source, linear in zip(sources, packed):
        plan = dispatch.classify_linear(linear)
        assert plan is not None and plan.packed and plan.bits == 4
        want = fast.oq_a8_qmm_t(
            qa,
            sa,
            ra,
            source.weight,
            mx.contiguous(source.scales.T),
            mx.contiguous(source.biases.T),
            4,
            act_mode,
            variant,
        )
        weight, scales, biases = dispatch._prepared_weights(linear)
        got = fast.oq_a8_qmm_t(
            qa, sa, ra, weight, scales, biases, 4, act_mode, variant, packed=True
        )
        mx.eval(want, got)
        np.testing.assert_array_equal(
            np.array(got.astype(mx.float32)), np.array(want.astype(mx.float32))
        )


@requires_kernels
@pytest.mark.parametrize("act_mode", [0, 1])
def test_stage_a_keeps_the_batch_rank(act_mode):
    """The op derives the output shape from Qa, so its rank has to survive.

    A flattened [B*S, K] Qa yields a [B*S, N] result. At B == 1 that
    broadcasts against the residual and hides.
    """
    fast = _kernels()
    rng = np.random.default_rng(31)
    x = mx.array(
        (rng.standard_normal((3, 128, 256)) * 0.5).astype(np.float32), mx.float16
    )
    mx.eval(x)
    qa, _, _ = fast.oq_a8_stage_a_v8(x, act_mode)
    mx.eval(qa)
    assert qa.shape == x.shape


@requires_kernels
@pytest.mark.parametrize("act_mode", [0, 1])
def test_stage_a_group_major_metadata_is_not_axis_reversed(act_mode):
    """Ra must come back [K/64, M], not [K/64, S, B].

    mx.transpose reverses every axis, so transposing a 3-D Ra in place gives
    the layout the kernel wants only when B == 1 and interleaves the
    sequences when it is not.
    """
    fast = _kernels()
    B, S, K = 3, 128, 256
    groups = K // GROUP_SIZE
    rng = np.random.default_rng(32)
    x = mx.array((rng.standard_normal((B, S, K)) * 0.5).astype(np.float32), mx.float16)
    mx.eval(x)

    _, sa_flat, ra_flat = fast.oq_a8_quantize(x, act_mode)
    _, sa, ra = fast.oq_a8_stage_a_v8(x, act_mode)
    mx.eval(sa_flat, ra_flat, sa, ra)

    expected = np.array(ra_flat).reshape(B * S, groups).T
    np.testing.assert_array_equal(np.array(ra), expected)
    if act_mode != 0:
        np.testing.assert_array_equal(
            np.array(sa), np.array(sa_flat).reshape(B * S, groups).T
        )


@requires_kernels
@pytest.mark.parametrize("B", [1, 2, 3])
def test_batched_prefill_keeps_sequences_independent(B):
    """End to end at B > 1: row b of the output depends only on row b of x."""
    fast = _kernels()
    S, K, N = 96, 256, 128
    rng = np.random.default_rng(33)
    w = mx.array(rng.standard_normal((N, K)).astype(np.float32) * 0.05, mx.float16)
    packed, scales, biases = mx.quantize(
        w, group_size=GROUP_SIZE, bits=4, mode="affine"
    )
    x = mx.array((rng.standard_normal((B, S, K)) * 0.5).astype(np.float32), mx.float16)
    mx.eval(packed, scales, biases, x)
    sc_t, bi_t = mx.contiguous(scales.T), mx.contiguous(biases.T)
    mx.eval(sc_t, bi_t)

    def run(inp):
        qa, sa, ra = fast.oq_a8_stage_a_v8(inp, 0)
        mx.eval(qa, sa, ra)
        out = fast.oq_a8_qmm_t(qa, sa, ra, packed, sc_t, bi_t, 4, 0, 806)
        mx.eval(out)
        return out

    got = run(x)
    assert got.shape == (B, S, N)
    ref = mx.quantized_matmul(
        x,
        packed,
        scales,
        biases,
        transpose=True,
        group_size=GROUP_SIZE,
        bits=4,
        mode="affine",
    )
    mx.eval(ref)
    g = np.array(got.astype(mx.float32))
    r = np.array(ref.astype(mx.float32))
    rel = np.abs(g - r).max() / max(np.abs(r).max(), 1e-6)
    assert rel < 0.06, f"rel={rel}"

    # Batching must not change any one sequence's result.
    for b in range(B):
        alone = np.array(run(x[b : b + 1]).astype(mx.float32))[0]
        np.testing.assert_array_equal(alone, g[b])


@requires_kernels
def test_short_sequences_stay_on_the_existing_path():
    """Below the token floor the Stage-A pass costs more than the GEMM saves."""
    from omlx.patches import qwen35_oq_a8 as dispatch

    linear = _opt_in(_quantized_linear(512, 128, 4), min_tokens=512)
    rng = np.random.default_rng(9)
    x = mx.array((rng.standard_normal((64, 512)) * 0.5).astype(np.float32), mx.float16)
    got = dispatch.oq_a8_linear(linear, x)
    mx.eval(got)
    # Bit-identical to the unrouted call, i.e. it really did fall through.
    np.testing.assert_array_equal(
        np.array(got.astype(mx.float32)), np.array(linear(x).astype(mx.float32))
    )


def test_patch_is_idempotent_and_reports_state():
    """Installing twice must not stack wrappers on the MLP class."""
    from omlx.patches import qwen35_oq_a8 as dispatch

    if not dispatch._kernels_available():
        pytest.skip("oQ A8 kernels unavailable")
    first = dispatch.apply_qwen35_oq_a8_patch()
    second = dispatch.apply_qwen35_oq_a8_patch()
    assert first == second


@requires_kernels
def test_patched_mlp_routes_and_falls_back(monkeypatch):
    """The installed wrapper must route long prompts and pass short ones through."""
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = _quantized_linear(256, 512, 4)
            self.up_proj = _quantized_linear(256, 512, 4)
            self.down_proj = _quantized_linear(512, 256, 4)

        def __call__(self, x, *args, **kwargs):
            return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

    monkeypatch.setattr(dispatch, "_SWIGLU", lambda g, u: nn.silu(g) * u)

    mlp = _opt_in(MLP(), min_tokens=128)
    orig = MLP.__call__
    MLP.__call__ = dispatch._make_patched_mlp(orig)

    rng = np.random.default_rng(21)
    long_x = mx.array(
        (rng.standard_normal((1, 256, 256)) * 0.4).astype(np.float32), mx.float16
    )
    short_x = long_x[:, :64, :]
    mx.eval(long_x, short_x)

    routed = mlp(long_x)
    plain = orig(mlp, long_x)
    mx.eval(routed, plain)
    rel = np.abs(
        np.array(routed.astype(mx.float32)) - np.array(plain.astype(mx.float32))
    ).max() / max(np.abs(np.array(plain.astype(mx.float32))).max(), 1e-6)
    assert 0.0 < rel < 0.08, f"long prompt did not route (rel={rel})"

    # Below the floor it must be the original computation, bit for bit.
    np.testing.assert_array_equal(
        np.array(mlp(short_x).astype(mx.float32)),
        np.array(orig(mlp, short_x).astype(mx.float32)),
    )

    # target_verify forwards are decode-shaped and must never route.
    np.testing.assert_array_equal(
        np.array(mlp(long_x, target_verify=True).astype(mx.float32)),
        np.array(plain.astype(mx.float32)),
    )


@requires_kernels
def test_patch_takes_its_configuration_from_the_caller():
    """The engine passes the model's settings in, rather than via the process
    environment, so two engines cannot silently reconfigure each other."""
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = _quantized_linear(512, 128, 4)

    model = Model()
    dispatch.apply_qwen35_oq_a8_patch(model, min_tokens=2048)
    config = dispatch._config_for(model.proj)
    assert config is not None and config.min_tokens == 2048

    # Q4 and Q5 are tuned independently; the tile is not a user setting.
    assert dispatch._variant_for_bits(4) == dispatch._DEFAULT_VARIANT_Q4
    assert dispatch._variant_for_bits(5) == dispatch._DEFAULT_VARIANT_Q5


@requires_kernels
def test_turning_the_setting_off_actually_stops_routing():
    """Enable, then reload with the setting off: the second model must run on
    MLX.

    The patch replaces Qwen3_5MLP.__call__ process-wide and never removes it,
    so a process-wide "enabled" flag would stay stuck on and the reload would
    keep routing at full speed. Opt-in lives on the model's own modules
    instead, and an untagged model falls straight through.
    """
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = _quantized_linear(512, 128, 4)

    on = Model()
    dispatch.apply_qwen35_oq_a8_patch(on, min_tokens=64)

    # A second model loaded with the setting off is simply never tagged.
    off = Model()

    rng = np.random.default_rng(11)
    x = mx.array((rng.standard_normal((128, 512)) * 0.5).astype(np.float32), mx.float16)

    routed = dispatch.oq_a8_linear(on.proj, x)
    untouched = dispatch.oq_a8_linear(off.proj, x)
    reference_on = on.proj(x)
    reference_off = off.proj(x)
    mx.eval(routed, untouched, reference_on, reference_off)

    # The tagged model reached the kernel: close, but not bit-identical.
    rel = np.abs(
        np.array(routed.astype(mx.float32)) - np.array(reference_on.astype(mx.float32))
    ).max() / max(np.abs(np.array(reference_on.astype(mx.float32))).max(), 1e-6)
    assert 0.0 < rel < 0.05, f"tagged model did not reach the kernel (rel={rel})"

    # The untagged one is bit-identical to MLX, i.e. it never left that path.
    np.testing.assert_array_equal(
        np.array(untouched.astype(mx.float32)),
        np.array(reference_off.astype(mx.float32)),
    )
    assert dispatch._config_for(off.proj) is None


@requires_kernels
def test_two_resident_models_keep_their_own_settings():
    """The wrapper is shared, so the floors must not be last-writer-wins."""
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = _quantized_linear(512, 128, 4)

    short, long = Model(), Model()
    dispatch.apply_qwen35_oq_a8_patch(short, min_tokens=64)
    dispatch.apply_qwen35_oq_a8_patch(long, min_tokens=4096)

    assert dispatch._config_for(short.proj).min_tokens == 64
    assert dispatch._config_for(long.proj).min_tokens == 4096


def test_opted_in_model_gets_its_own_ssd_numerics_revision():
    """KV computed from INT8 activations must not serve a model with A8 off."""
    import mlx.nn as nn

    from omlx.cache.paged_ssd_cache import numerics_revision_for_model

    assert numerics_revision_for_model(nn.Linear(4, 4)) is None
    assert numerics_revision_for_model(_opt_in(nn.Linear(4, 4))) == "a8-prefill-1"


@pytest.mark.parametrize("batch", [1, 4])
def test_single_token_mlp_stays_on_decode_when_floor_is_one(monkeypatch, batch):
    from unittest.mock import Mock

    from omlx.patches import qwen35_oq_a8 as dispatch

    route = Mock(side_effect=AssertionError("decode entered A8"))
    monkeypatch.setattr(dispatch, "oq_a8_mlp", route)
    original = Mock(return_value="decode")
    wrapped = dispatch._make_patched_mlp(original)
    x = mx.zeros((batch, 1, 64), dtype=mx.float16)
    assert wrapped(_opt_in(SimpleNamespace(), min_tokens=1), x) == "decode"
    route.assert_not_called()
    assert not dispatch._shape_eligible(x, dispatch.OqA8Config(min_tokens=1))
    assert dispatch._shape_eligible(
        mx.zeros((1, 128, 64), mx.float16), dispatch.OqA8Config(min_tokens=1)
    )


@requires_kernels
@pytest.mark.parametrize("variant", [-1, 0, 6, 200, 206, 799, 807])
def test_native_qmm_rejects_removed_variants(variant):
    fast = _kernels()
    packed, scales, biases = make_quantized(128, 128, 4, seed=20)
    x = mx.ones((128, 128), dtype=mx.float16)
    with pytest.raises(ValueError, match="variant"):
        fast.oq_a8_linear(x, packed, scales, biases, 4, variant=variant)


def test_mlp_routing_errors_are_not_silently_ignored(monkeypatch):
    from unittest.mock import Mock

    from omlx.patches import qwen35_oq_a8 as dispatch

    route = Mock(side_effect=RuntimeError("kernel failure"))
    original = Mock()
    monkeypatch.setattr(dispatch, "oq_a8_mlp", route)
    monkeypatch.setattr(dispatch, "_SWIGLU", lambda g, u: g * u)
    wrapped = dispatch._make_patched_mlp(original)
    with pytest.raises(RuntimeError, match="kernel failure"):
        wrapped(_opt_in(SimpleNamespace()), mx.ones((1, 128, 64), dtype=mx.float16))
    original.assert_not_called()


# --------------------------------------------------------------------------
# Q8 (GS64 affine, W8A8)
# --------------------------------------------------------------------------


@requires_kernels
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("act_mode", [0, 1])
@pytest.mark.parametrize("m", [1, 17, 130])
def test_q8_gemm_matches_affine_reference(dtype, act_mode, m):
    """Centered GEMM vs unsigned-code affine reference; m covers partial tiles."""
    fast = _kernels()
    n, k = 128, 192
    rng = np.random.default_rng(m + 5 * act_mode)
    codes = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    codes[:, :4] = [0, 127, 128, 255]
    codes[0, :] = 255
    codes[1, :] = 0
    packed = mx.array(codes.view("<u4").copy())
    scales = mx.array(
        rng.uniform(0.001, 0.02, (n, k // GROUP_SIZE)).astype(np.float32), dtype
    )
    biases = mx.array(
        rng.uniform(-1, 1, (n, k // GROUP_SIZE)).astype(np.float32), dtype
    )
    x = mx.array((rng.standard_normal((m, k)) * 0.5).astype(np.float32), dtype)

    qa, sa, ra = fast.oq_a8_quantize(x, act_mode)
    qa8, sa8, ra8 = fast.oq_a8_stage_a_natural(x, act_mode)
    got = fast.oq_a8_qmm_t(qa8, sa8, ra8, packed, scales, biases, 8, act_mode)
    mx.eval(qa, sa, ra, got)

    expected = affine_reference(
        np.array(qa).astype(np.int64),
        np.array(sa).reshape(m) if act_mode == 0 else np.array(sa),
        np.array(ra).astype(np.int64),
        codes.astype(np.int64),
        np.array(scales.astype(mx.float32)),
        np.array(biases.astype(mx.float32)),
        act_mode,
    )
    got_np = np.array(got.astype(mx.float32))
    assert np.isfinite(got_np).all()
    # Only the output rounding to `dtype` separates the two.
    np.testing.assert_allclose(got_np, expected, rtol=2**-7, atol=2e-3)


@requires_kernels
@pytest.mark.parametrize("act_mode", [0, 1])
def test_stage_a_natural_keeps_k_order(act_mode):
    """Qa stays unpermuted; only the group metadata is transposed."""
    fast = _kernels()
    rng = np.random.default_rng(3)
    x = mx.array((rng.standard_normal((40, 256)) * 0.7).astype(np.float32), mx.float16)
    qa, sa, ra = fast.oq_a8_quantize(x, act_mode)
    qn, sn, rn = fast.oq_a8_stage_a_natural(x, act_mode)
    mx.eval(qa, sa, ra, qn, sn, rn)

    np.testing.assert_array_equal(np.array(qn), np.array(qa))
    np.testing.assert_array_equal(np.array(rn), np.array(ra).T)
    expected_sa = np.array(sa).T if act_mode else np.array(sa)
    np.testing.assert_array_equal(np.array(sn), expected_sa)


@requires_kernels
@pytest.mark.parametrize("act_mode", [0, 1])
def test_q8_tiles_and_token_paths_agree(act_mode):
    """Rows are independent, so both tiles and both Ra/Sa load paths agree."""
    fast = _kernels()
    k, n = 256, 128
    packed, scales, biases = make_quantized(n, k, 8, seed=5)
    rows = np.random.default_rng(5).standard_normal((1028, k)).astype(np.float32)

    def run(m):
        x = mx.array(rows[:m] * 0.5, mx.float16)
        qa, sa, ra = fast.oq_a8_stage_a_natural(x, act_mode)
        y = fast.oq_a8_qmm_t(qa, sa, ra, packed, scales, biases, 8, act_mode)
        return np.array(y.astype(mx.float32)).view(np.uint32)

    narrow = run(1028)  # (1,2) tile, vector Ra/Sa loads
    np.testing.assert_array_equal(run(1027), narrow[:1027])  # scalar loads
    np.testing.assert_array_equal(run(1024), narrow[:1024])  # (2,2) tile
    np.testing.assert_array_equal(run(130), narrow[:130])  # (2,2), scalar


@requires_kernels
def test_q8_rejects_group_major_metadata():
    """Q8 metadata is [N, K/64]; a transposed copy is a shape error."""
    fast = _kernels()
    packed, scales, biases = make_quantized(128, 256, 8)
    qa, sa, ra = fast.oq_a8_stage_a_natural(mx.zeros((64, 256), mx.float16), 0)
    with pytest.raises(ValueError, match="incompatible"):
        fast.oq_a8_qmm_t(
            qa, sa, ra, packed, mx.contiguous(scales.T), mx.contiguous(biases.T), 8
        )


@pytest.mark.parametrize("bits", [4, 5, 8])
def test_apply_plan_metadata_layout(monkeypatch, bits):
    """Q8 gets the layer's own [N, K/64] arrays; Q4/Q5 a cached [K/64, N] copy."""
    from omlx.custom_kernels.quant_matmul import fast
    from omlx.patches import qwen35_oq_a8 as dispatch

    calls = []
    monkeypatch.setattr(fast, "oq_a8_available", lambda: True)
    monkeypatch.setattr(
        fast,
        "oq_a8_qmm_t",
        lambda qa, sa, ra, w, s, b, *args, **kwargs: calls.append((w, s, b)),
    )
    linear = _quantized_linear(256, 128, bits)
    plan = dispatch.classify_linear(linear)
    stage = dispatch.StageA(
        qa=mx.zeros((4, 256), mx.int8),
        sa=mx.zeros((4,), mx.float32),
        ra=mx.zeros((4, 4), mx.int16),
        act_mode=plan.act_mode,
        layout=plan.layout,
    )
    dispatch.apply_plan(linear, stage, plan)

    weight, scales, biases = calls[0]
    cached = getattr(linear, dispatch._PREPARED_ATTR, None)
    assert weight is linear.weight
    if bits == 8:
        assert scales is linear.scales and biases is linear.biases
        assert cached is None
    else:
        assert scales.shape == biases.shape == (256 // GROUP_SIZE, 128)
        assert cached is not None


def _stage_a_counter(monkeypatch, dispatch):
    calls = []
    real = dispatch.stage_a

    def counting(x, act_mode=dispatch._ACT_MODE_ROW, layout=dispatch._LAYOUT_V8):
        calls.append(layout)
        return real(x, act_mode, layout)

    monkeypatch.setattr(dispatch, "stage_a", counting)
    return calls


@requires_kernels
def test_q8_mlp_shares_gate_up_stage_a(monkeypatch):
    """gate and up share one Stage A; down quantizes its own input."""
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = _quantized_linear(256, 512, 8)
            self.up_proj = _quantized_linear(256, 512, 8)
            self.down_proj = _quantized_linear(512, 256, 8)

    mlp = _opt_in(MLP())
    x = mx.array(
        (np.random.default_rng(1).standard_normal((1, 256, 256)) * 0.4).astype(
            np.float32
        ),
        mx.float16,
    )
    calls = _stage_a_counter(monkeypatch, dispatch)
    got = dispatch.oq_a8_mlp(mlp, x, lambda g, u: nn.silu(g) * u)
    ref = mlp.down_proj(nn.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    mx.eval(got, ref)

    assert calls == [dispatch._LAYOUT_NATURAL] * 2
    g, r = (np.array(a.astype(mx.float32)) for a in (got, ref))
    rel = np.abs(g - r).max() / max(np.abs(r).max(), 1e-6)
    assert 0.0 < rel < 0.08, f"Q8 MLP did not reach the kernel (rel={rel})"


@requires_kernels
def test_mixed_stage_a_layouts_decline():
    """A Q4 gate with a Q8 up would need two Stage A layouts."""
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = _quantized_linear(256, 512, 4)
            self.up_proj = _quantized_linear(256, 512, 8)
            self.down_proj = _quantized_linear(512, 256, 4)

    x = mx.zeros((1, 256, 256), mx.float16)
    assert dispatch.oq_a8_mlp(_opt_in(MLP()), x, lambda g, u: g * u) is None


class _GatedDeltaProjections:
    def __init__(self, bits, n_ab=64):
        self.in_proj_qkv = _quantized_linear(256, 384, bits)
        self.in_proj_z = _quantized_linear(256, 128, bits)
        self.in_proj_a = _quantized_linear(256, n_ab, bits)
        self.in_proj_b = _quantized_linear(256, n_ab, bits)


@requires_kernels
@pytest.mark.parametrize("bits", [4, 8])
def test_gdn_projections_share_one_stage_a(monkeypatch, bits):
    from omlx.patches import qwen35_oq_a8 as dispatch

    gdn = _opt_in(_GatedDeltaProjections(bits))
    x = mx.array(
        (np.random.default_rng(2).standard_normal((1, 256, 256)) * 0.5).astype(
            np.float32
        ),
        mx.float16,
    )
    calls = _stage_a_counter(monkeypatch, dispatch)
    out = dispatch.oq_a8_gdn_projections(gdn, x)
    mx.eval(*out.values())

    assert calls == [dispatch._layout_for_bits(bits)]
    assert set(out) == {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"}
    for name, got in out.items():
        ref = np.array(getattr(gdn, name)(x).astype(mx.float32))
        rel = np.abs(np.array(got.astype(mx.float32)) - ref).max() / np.abs(ref).max()
        assert 0.0 < rel < 0.08


@requires_kernels
@pytest.mark.parametrize("bits", [4, 8])
def test_gdn_untiled_projection_declines_block(bits):
    """in_proj_a/b are N=48 on Qwen3.8; one untiled projection declines the block."""
    from omlx.patches import qwen35_oq_a8 as dispatch

    gdn = _opt_in(_GatedDeltaProjections(bits, n_ab=48))
    assert (
        dispatch.oq_a8_gdn_projections(gdn, mx.zeros((1, 256, 256), mx.float16)) is None
    )


@requires_kernels
def test_stage_a_layout_mismatch_raises():
    from omlx.patches import qwen35_oq_a8 as dispatch

    linear = _quantized_linear(256, 128, 8)
    plan = dispatch.classify_linear(linear)
    x = mx.zeros((64, 256), mx.float16)
    with pytest.raises(ValueError, match="layout"):
        dispatch.apply_plan(linear, dispatch.stage_a(x, plan.act_mode, 0), plan)


@requires_kernels
def test_q8_standalone_route_and_fallback():
    from omlx.patches import qwen35_oq_a8 as dispatch

    linear = _opt_in(_quantized_linear(256, 128, 8), min_tokens=128)
    x = mx.array(
        (np.random.default_rng(6).standard_normal((1, 256, 256)) * 0.5).astype(
            np.float32
        ),
        mx.float16,
    )
    routed = dispatch._prefill_linear_backend(linear, x)
    mx.eval(routed)
    ref = linear(x)
    assert routed is not None
    rel = (
        np.abs(
            np.array(routed.astype(mx.float32)) - np.array(ref.astype(mx.float32))
        ).max()
        / np.abs(np.array(ref.astype(mx.float32))).max()
    )
    assert 0.0 < rel < 0.05

    assert dispatch._prefill_linear_backend(linear, x[:, :64]) is None
    assert dispatch._prefill_linear_backend(linear, x[:, :1]) is None


@requires_kernels
@pytest.mark.parametrize("head_bits, tagged", [(8, False), (4, True)])
def test_q8_mtp_head_not_tagged(head_bits, tagged):
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = _quantized_linear(256, 128, 8)
            self.head = nn.Module()
            self.head.proj = _quantized_linear(256, 128, head_bits)

        def get_mtp_module(self):
            return self.head

    model = Model()
    dispatch.apply_qwen35_oq_a8_patch(model, min_tokens=64)

    assert dispatch._config_for(model.proj) is not None
    assert (dispatch._config_for(model.head.proj) is not None) is tagged


def _stock_affine_format_linear(bits=4, out_dim=128, in_dim=256):
    import json
    import pathlib
    import tempfile

    from omlx.formats import testing as fmt_testing
    from omlx.formats.linear import FormatLinear
    from omlx.patches.format_load import loading

    directory = pathlib.Path(tempfile.mkdtemp())
    (directory / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "omlx_quant_formats": {"affine": {}}})
    )
    mx.random.seed(17)
    weight = (mx.random.normal((out_dim, in_dim)) * 0.1).astype(mx.float16)
    arrays = dict(zip(("weight", "scales", "biases"), mx.quantize(weight, bits=bits)))
    with fmt_testing.verified("qwen4_exp"), loading(directory):
        return FormatLinear("affine", bits, GROUP_SIZE, in_dim, arrays)


@requires_kernels
def test_stock_affine_format_layer_classifies_like_mlxs_layer():
    from omlx.patches import qwen35_oq_a8 as dispatch

    layer = _stock_affine_format_linear(4)
    plan = dispatch.classify_linear(layer)
    assert plan is not None, "a stock affine FormatLinear must take the A8 plan"
    assert (plan.bits, plan.group_size, plan.kernel) == (4, 64, "q4a8_g64")


@requires_kernels
def test_a8_plan_on_the_format_layer_is_bitwise_the_mlx_layer_plan():
    """83ee1d7b ran these projections as QuantizedLinear; the format layer
    carries the same tensors, so the same A8 kernel must produce the same
    bits."""
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    fmt = _stock_affine_format_linear(4)
    quantized = nn.QuantizedLinear(256, 128, bias=False, group_size=64, bits=4)
    quantized.set_dtype(mx.float16)
    for name in ("weight", "scales", "biases"):
        setattr(quantized, name, fmt[name])
    plans = (dispatch.classify_linear(fmt), dispatch.classify_linear(quantized))
    assert plans[0] is not None and plans[1] is not None
    mx.random.seed(23)
    x = (mx.random.normal((32, 256)) * 0.5).astype(mx.float16)
    outs = [
        dispatch.apply_plan(
            layer, dispatch.stage_a(x, plan.act_mode, plan.layout), plan
        )
        for layer, plan in zip((fmt, quantized), plans)
    ]
    assert mx.array_equal(outs[0].view(mx.uint16), outs[1].view(mx.uint16))


@requires_kernels
def test_apply_allows_qwen4_n48_gdn(monkeypatch):
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    class Gdn(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj_qkv = _quantized_linear(256, 256, 4)
            self.in_proj_z = _quantized_linear(256, 128, 4)
            self.in_proj_b = _quantized_linear(256, 48, 5)
            self.in_proj_a = _quantized_linear(256, 48, 5)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = _quantized_linear(256, 128, 4)
            self.up_proj = _quantized_linear(256, 128, 4)
            self.down_proj = _quantized_linear(128, 256, 4)
            self.linear_attn = Gdn()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Block()

    Model.__module__ = "mlx_vlm.models.qwen4_exp.language"
    Gdn.__module__ = "mlx_vlm.models.qwen4_exp.language"
    Block.__module__ = "mlx_vlm.models.qwen4_exp.language"
    monkeypatch.setattr(dispatch, "_kernels_available", lambda: True)
    monkeypatch.setattr(dispatch, "_mtp_module_ids", lambda model: set())
    dispatch.apply_qwen35_oq_a8_patch(Model())


def test_refuses_a_dense_projection_the_a8_kernels_cannot_take(monkeypatch):
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = _quantized_linear(256, 128, 4)
            self.up_proj = _quantized_linear(256, 128, 4)
            down = _stock_affine_format_linear(4, out_dim=256, in_dim=128)
            object.__setattr__(down, "fmt", object())
            self.down_proj = down

    Block.__module__ = "mlx_vlm.models.qwen4_exp.language"
    monkeypatch.setattr(dispatch, "_mtp_module_ids", lambda model: set())
    with pytest.raises(ValueError, match="down_proj"):
        dispatch._refuse_unclassified(Block())


def test_classified_dense_projections_load(monkeypatch):
    from omlx.patches import qwen35_oq_a8 as dispatch

    class Block:
        gate_proj = _quantized_linear(256, 128, 4)
        up_proj = _quantized_linear(256, 128, 4)
        down_proj = _quantized_linear(128, 256, 8)

        def named_modules(self):
            return [("", self)]

    Block.__module__ = "mlx_vlm.models.qwen4_exp.language"
    plan = dispatch.OqA8Plan(bits=4, group_size=64, act_mode=0, variant=800)
    calls = []

    def stub_classify(linear):
        calls.append(linear)
        return plan

    monkeypatch.setattr(dispatch, "_mtp_module_ids", lambda model: set())
    monkeypatch.setattr(dispatch, "classify_linear", stub_classify)
    dispatch._refuse_unclassified(Block())
    assert len(calls) == 3


def test_apply_refusal_leaves_model_untagged(monkeypatch):
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = _quantized_linear(256, 128, 4)
            self.up_proj = _quantized_linear(256, 128, 4)
            down = _stock_affine_format_linear(4, out_dim=256, in_dim=128)
            object.__setattr__(down, "fmt", object())
            self.down_proj = down

    Block.__module__ = "mlx_vlm.models.qwen4_exp.language"

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Block()

    Model.__module__ = "mlx_vlm.models.qwen4_exp.language"

    model = Model()
    monkeypatch.setattr(dispatch, "_kernels_available", lambda: True)
    monkeypatch.setattr(dispatch, "_mtp_module_ids", lambda model: set())
    with pytest.raises(ValueError, match="down_proj"):
        dispatch.apply_qwen35_oq_a8_patch(model)
    assert getattr(model.mlp, dispatch._CONFIG_ATTR, None) is None
    assert getattr(model.mlp.gate_proj, dispatch._CONFIG_ATTR, None) is None


def test_refuses_non_affine_format_and_unaligned_packed_linear(monkeypatch):
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch
    from omlx.patches.qwen35_packed_linear import PackedLinear

    layer = _stock_affine_format_linear(4)
    object.__setattr__(layer, "fmt", object())  # any non-AffineFormat value

    class Packed(PackedLinear):
        output_dims = 96

        def __init__(self):
            nn.Module.__init__(self)
            object.__setattr__(self, "packed_weight", layer["weight"])
            object.__setattr__(self, "packed_scales", layer["scales"])
            object.__setattr__(self, "packed_biases", layer["biases"])

    class Block:
        gate_proj = _quantized_linear(256, 128, 4)
        up_proj = _quantized_linear(256, 128, 4)

        def __init__(self, down):
            self.down_proj = down

        def named_modules(self):
            return [("", self)]

    Block.__module__ = "mlx_vlm.models.qwen4_exp.language"
    monkeypatch.setattr(dispatch, "_mtp_module_ids", lambda model: set())
    for down in (layer, Packed()):
        with pytest.raises(ValueError, match="down_proj"):
            dispatch._refuse_unclassified(Block(down))

    # A 64-aligned PackedLinear (not 128-aligned) is not refused as unaligned.
    class Packed64(Packed):
        output_dims = 64

    plan = dispatch.OqA8Plan(bits=4, group_size=64, act_mode=0, variant=800)
    monkeypatch.setattr(
        dispatch,
        "classify_linear",
        lambda lin: None if isinstance(lin, PackedLinear) else plan,
    )
    dispatch._refuse_unclassified(Block(Packed64()))


def test_q8_stock_affine_format_mtp_head_not_tagged(monkeypatch):
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = _stock_affine_format_linear(8)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = _quantized_linear(256, 128, 4)
            self.head = Head()

        def get_mtp_module(self):
            return self.head

    model = Model()
    tagged = dispatch._tag_modules(model, dispatch.OqA8Config(min_tokens=128))
    assert tagged >= 1
    assert getattr(model.head.proj, dispatch._CONFIG_ATTR, None) is None


def test_refuses_non_affine_attention_projection(monkeypatch):
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = _quantized_linear(256, 128, 4)
            self.k_proj = _quantized_linear(256, 128, 4)
            self.v_proj = _quantized_linear(256, 128, 4)
            o_proj = _stock_affine_format_linear(4, out_dim=256, in_dim=128)
            object.__setattr__(o_proj, "fmt", object())
            self.o_proj = o_proj

    Attn.__module__ = "mlx_vlm.models.qwen4_exp.language"
    monkeypatch.setattr(dispatch, "_mtp_module_ids", lambda model: set())
    with pytest.raises(ValueError, match="o_proj.*output_dims=256"):
        dispatch._refuse_unclassified(Attn())


def test_zero_classify_allows_routed_a8_with_q6_dense(monkeypatch):
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwiGLU

    from omlx.patches import qwen35_oq_a8 as dispatch

    class DenseBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = _quantized_linear(256, 128, 6)
            self.up_proj = _quantized_linear(256, 128, 6)
            self.down_proj = _quantized_linear(128, 256, 6)

    class RoutedSwitchGLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = QuantizedSwitchLinear(
                256, 256, 4, bias=False, group_size=64, bits=4
            )
            self.down_proj = QuantizedSwitchLinear(
                128, 256, 4, bias=False, group_size=64, bits=4
            )
            self.activation = SwiGLU()

    class Model(nn.Module):
        def __init__(self, with_routed):
            super().__init__()
            self.mlp = DenseBlock()
            if with_routed:
                self.switch_mlp = RoutedSwitchGLU()

    DenseBlock.__module__ = "mlx_vlm.models.qwen4_exp.language"
    RoutedSwitchGLU.__module__ = "mlx_vlm.models.qwen4_exp.language"
    Model.__module__ = "mlx_vlm.models.qwen4_exp.language"
    monkeypatch.setattr(dispatch, "_mtp_module_ids", lambda model: set())
    monkeypatch.setattr(dispatch, "classify_linear", lambda linear: None)

    # With eligible Q4 GS64 routed SwitchGLU, A8 opt-in is live -> passes.
    dispatch._refuse_unclassified(Model(with_routed=True))

    # Without routed A8, zero classified dense projections -> refuses.
    with pytest.raises(ValueError, match="no Qwen projection"):
        dispatch._refuse_unclassified(Model(with_routed=False))


def test_refuse_unclassified_skips_qwen4_vision_namespace(monkeypatch):
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    class VisionBlock(nn.Module):
        def __init__(self):
            super().__init__()
            out = _stock_affine_format_linear(4, out_dim=256, in_dim=128)
            object.__setattr__(out, "fmt", object())
            self.out_proj = out

    VisionBlock.__module__ = "mlx_vlm.models.qwen4_exp.vision"
    monkeypatch.setattr(dispatch, "_mtp_module_ids", lambda model: set())
    dispatch._refuse_unclassified(VisionBlock())


def test_refuse_unclassified_covers_qwen35_and_skips_mtp(monkeypatch):
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    class Qwen35Block(nn.Module):
        def __init__(self):
            super().__init__()
            out = _stock_affine_format_linear(4, out_dim=256, in_dim=128)
            object.__setattr__(out, "fmt", object())
            self.out_proj = out

    Qwen35Block.__module__ = "mlx_vlm.models.qwen3_5.language"
    block = Qwen35Block()
    monkeypatch.setattr(dispatch, "_mtp_module_ids", lambda model: set())
    with pytest.raises(ValueError, match="out_proj"):
        dispatch._refuse_unclassified(block)

    # Same module inside MTP draft head is excluded.
    monkeypatch.setattr(dispatch, "_mtp_module_ids", lambda model: {id(block)})
    dispatch._refuse_unclassified(block)


def test_engine_loaders_propagate_oq_a8_value_error(monkeypatch):
    from types import SimpleNamespace

    from omlx.engine.batched import BatchedEngine
    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.patches import qwen35_oq_a8 as dispatch

    def raise_value_error(model, min_tokens=None):
        raise ValueError("oq_a8: refused")

    def raise_runtime_error(model, min_tokens=None):
        raise RuntimeError("transient")

    vlm_engine = VLMBatchedEngine.__new__(VLMBatchedEngine)
    vlm_engine._vlm_model = object()
    vlm_engine._model_settings = SimpleNamespace(a8_prefill_min_tokens=128)

    batched_engine = BatchedEngine.__new__(BatchedEngine)
    batched_engine._model = object()
    batched_engine._model_settings = SimpleNamespace(a8_prefill_min_tokens=128)

    monkeypatch.setattr(dispatch, "apply_qwen35_oq_a8_patch", raise_value_error)
    with pytest.raises(ValueError, match="oq_a8: refused"):
        vlm_engine._apply_oq_a8_prefill()
    with pytest.raises(ValueError, match="oq_a8: refused"):
        batched_engine._apply_oq_a8_prefill()

    monkeypatch.setattr(dispatch, "apply_qwen35_oq_a8_patch", raise_runtime_error)
    vlm_engine._apply_oq_a8_prefill()
    batched_engine._apply_oq_a8_prefill()


def test_has_routed_a8_guards(monkeypatch):
    import mlx.nn as nn

    from omlx.patches import qwen35_oq_a8 as dispatch

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a_live_block = nn.Module()
            self.z_mtp_block = nn.Module()

    model = Model()
    blocks = (model.z_mtp_block, model.a_live_block)
    monkeypatch.setattr(
        dispatch, "_is_fused_routed_glu", lambda module: module in blocks
    )
    monkeypatch.setattr(dispatch, "_routed_a8_eligible", lambda module, act: True)
    monkeypatch.setattr(dispatch, "_is_measured_family", lambda m: False)
    assert dispatch._has_routed_a8(model, set()) is False
    monkeypatch.setattr(dispatch, "_is_measured_family", lambda m: True)
    assert dispatch._has_routed_a8(model, {id(model.z_mtp_block)}) is True
    assert dispatch._has_routed_a8(model, {id(model.a_live_block)}) is True
    assert (
        dispatch._has_routed_a8(model, {id(model.z_mtp_block), id(model.a_live_block)})
        is False
    )
    monkeypatch.setattr(dispatch, "_routed_a8_eligible", lambda module, act: False)
    assert dispatch._has_routed_a8(model, set()) is False


def test_refuse_unclassified_rejects_unquantized_qwen_model(tmp_path, monkeypatch):
    import json

    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwiGLU

    from omlx.formats import testing as fmt_testing
    from omlx.formats.linear import FormatSwitchLinear
    from omlx.patches import qwen35_oq_a8 as dispatch
    from omlx.patches.format_load import loading

    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(256, 128, bias=False)
            self.up_proj = nn.Linear(256, 128, bias=False)
            self.down_proj = nn.Linear(128, 256, bias=False)

    Mlp.__module__ = "mlx_vlm.models.qwen4_exp.language"

    class Model(nn.Module):
        def __init__(self, switch_mlp=None):
            super().__init__()
            self.mlp = Mlp()
            if switch_mlp is not None:
                self.switch_mlp = switch_mlp

    Model.__module__ = "mlx_vlm.models.qwen4_exp.language"
    model = Model()
    monkeypatch.setattr(dispatch, "_mtp_module_ids", lambda m: set())
    with pytest.raises(ValueError, match="no Qwen projection"):
        dispatch._refuse_unclassified(model)

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "omlx_quant_formats": {"affine": {}}})
    )
    stacked = mx.random.normal((2, 16, 256)).astype(mx.float16)
    parts = [mx.quantize(stacked[i], bits=4, group_size=64) for i in range(2)]
    tensors = {
        "weight": mx.stack([part[0] for part in parts]),
        "scales": mx.stack([part[1] for part in parts]),
        "biases": mx.stack([part[2] for part in parts]),
    }
    with fmt_testing.verified("qwen4_exp"), loading(tmp_path):
        gate_up = FormatSwitchLinear("affine", 4, 64, 256, tensors)
        down = FormatSwitchLinear("affine", 4, 64, 256, tensors)

    class RoutedSwitchGLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = gate_up
            self.down_proj = down
            self.activation = SwiGLU()

    RoutedSwitchGLU.__module__ = "mlx_vlm.models.switch_layers"
    dispatch._refuse_unclassified(Model(switch_mlp=RoutedSwitchGLU()))


def test_refuse_unclassified_rejects_non_affine_switch_experts(tmp_path, monkeypatch):
    import json

    import mlx.nn as nn

    from omlx.formats import testing as fmt_testing
    from omlx.formats.linear import FormatSwitchLinear
    from omlx.patches import qwen35_oq_a8 as dispatch
    from omlx.patches.format_load import loading

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "omlx_quant_formats": {"affine": {}}})
    )

    stacked = mx.random.normal((2, 16, 256)).astype(mx.float16)
    parts = [mx.quantize(stacked[i], bits=4, group_size=64) for i in range(2)]
    tensors = {
        "weight": mx.stack([part[0] for part in parts]),
        "scales": mx.stack([part[1] for part in parts]),
        "biases": mx.stack([part[2] for part in parts]),
    }
    with fmt_testing.verified("qwen4_exp"), loading(tmp_path):
        switch = FormatSwitchLinear("affine", 4, 64, 256, tensors)
    object.__setattr__(switch, "fmt", object())

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = switch

    Experts.__module__ = "mlx_lm.models.switch_layers"

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = Experts()

    Model.__module__ = "mlx_vlm.models.qwen4_exp.language"
    model = Model()
    monkeypatch.setattr(dispatch, "_mtp_module_ids", lambda m: set())
    with pytest.raises(ValueError, match="gate_up_proj"):
        dispatch._refuse_unclassified(model)


def test_refuse_unclassified_real_switch_namespace(tmp_path):
    import json

    import mlx.nn as nn

    from omlx.formats import QuantFormat
    from omlx.formats import testing as fmt_testing
    from omlx.formats.linear import FormatSwitchLinear
    from omlx.patches import qwen35_oq_a8 as dispatch
    from omlx.patches.format_load import loading

    class Plugin(QuantFormat):
        mode = "plugin_sw_test"
        codes_name = "weight"
        tensors = ("scales", "biases")
        bits = (4,)
        group_sizes = (64,)

        def layout(self, k, n, bits, group_size):
            return {
                "weight": ((n, k * bits // 32), mx.uint32),
                "scales": ((n, k // group_size), None),
                "biases": ((n, k // group_size), None),
            }

        def decode(self, tensors, bits, group_size):
            return mx.dequantize(
                tensors["weight"],
                tensors["scales"],
                tensors["biases"],
                group_size=group_size,
                bits=bits,
                mode="affine",
            )

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "omlx_quant_formats": {"affine": {}, Plugin.mode: {}},
            }
        )
    )

    def switch(fmt):
        stacked = mx.random.normal((2, 16, 256)).astype(mx.float16)
        parts = [mx.quantize(stacked[i], bits=4, group_size=64) for i in range(2)]
        tensors = {
            "weight": mx.stack([part[0] for part in parts]),
            "scales": mx.stack([part[1] for part in parts]),
            "biases": mx.stack([part[2] for part in parts]),
        }
        with fmt_testing.verified("qwen4_exp"), loading(tmp_path):
            return FormatSwitchLinear(fmt, 4, 64, 256, tensors)

    class Experts(nn.Module):
        def __init__(self, up, down):
            super().__init__()
            self.gate_up_proj = up
            self.down_proj = down

    Experts.__module__ = "mlx_lm.models.switch_layers"

    class Model(nn.Module):
        def __init__(self, experts):
            super().__init__()
            self.switch_mlp = experts

    # Top-level VLM wrapper: measured family, outside _QWEN_PREFIXES.
    Model.__module__ = "mlx_vlm.models.qwen4_exp.qwen4_exp"

    with pytest.raises(ValueError, match="gate_up_proj.*plugin_sw_test"):
        dispatch._refuse_unclassified(
            Model(Experts(switch(Plugin()), switch("affine")))
        )
    with pytest.raises(ValueError, match="down_proj.*plugin_sw_test"):
        dispatch._refuse_unclassified(
            Model(Experts(switch("affine"), switch(Plugin())))
        )
    dispatch._refuse_unclassified(Model(Experts(switch("affine"), switch("affine"))))

    class MtpModel(nn.Module):
        def __init__(self, experts):
            super().__init__()
            self.mtp = experts

        def get_mtp_module(self):
            return self.mtp

    MtpModel.__module__ = "mlx_vlm.models.qwen4_exp.qwen4_exp"
    dispatch._refuse_unclassified(MtpModel(Experts(switch(Plugin()), switch("affine"))))

    # The MTP head is skipped, not the rest of the scan: a live stack after
    # it must still refuse (a `break` in place of the skip loads it silently).
    class MtpThenLiveModel(nn.Module):
        def __init__(self, mtp, live):
            super().__init__()
            # MLX's named_modules walks a module's children last-in first-out:
            # registering live before mtp visits the whole MTP head first.
            self.live = live
            self.mtp = mtp

        def get_mtp_module(self):
            return self.mtp

    MtpThenLiveModel.__module__ = "mlx_vlm.models.qwen4_exp.qwen4_exp"
    model = MtpThenLiveModel(
        Experts(switch(Plugin()), switch("affine")),
        Experts(switch(Plugin()), switch("affine")),
    )
    visited = [id(module) for _, module in model.named_modules()]
    head = [visited.index(id(module)) for module in model.mtp.modules()]
    live = [visited.index(id(module)) for module in model.live.modules()]
    assert max(head) < min(live)
    with pytest.raises(ValueError, match="gate_up_proj.*plugin_sw_test"):
        dispatch._refuse_unclassified(model)


def test_refuse_unclassified_unfused_switch_namespace(tmp_path):
    import json

    import mlx.nn as nn

    from omlx.formats import QuantFormat
    from omlx.formats import testing as fmt_testing
    from omlx.formats.linear import FormatSwitchLinear
    from omlx.patches import qwen35_oq_a8 as dispatch
    from omlx.patches.format_load import loading

    class Plugin(QuantFormat):
        mode = "plugin_sw_unfused_test"
        codes_name = "weight"
        tensors = ("scales", "biases")
        bits = (4,)
        group_sizes = (64,)

        def layout(self, k, n, bits, group_size):
            return None

        def decode(self, tensors, bits, group_size):
            return mx.dequantize(
                tensors["weight"],
                tensors["scales"],
                tensors["biases"],
                group_size=group_size,
                bits=bits,
                mode="affine",
            )

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "omlx_quant_formats": {"affine": {}, Plugin.mode: {}},
            }
        )
    )
    stacked = mx.random.normal((2, 16, 256)).astype(mx.float16)
    parts = [mx.quantize(stacked[i], bits=4, group_size=64) for i in range(2)]
    tensors = {
        "weight": mx.stack([part[0] for part in parts]),
        "scales": mx.stack([part[1] for part in parts]),
        "biases": mx.stack([part[2] for part in parts]),
    }
    with fmt_testing.verified("qwen4_exp"), loading(tmp_path):
        plugin = FormatSwitchLinear(Plugin(), 4, 64, 256, tensors)
        affine = FormatSwitchLinear("affine", 4, 64, 256, tensors)

    class Experts(nn.Module):
        def __init__(self, gate, up, down):
            super().__init__()
            self.gate_proj = gate
            self.up_proj = up
            self.down_proj = down

    Experts.__module__ = "mlx_lm.models.switch_layers"

    class Model(nn.Module):
        def __init__(self, experts):
            super().__init__()
            self.switch_mlp = experts

    Model.__module__ = "mlx_vlm.models.qwen4_exp.language"

    with pytest.raises(ValueError, match="gate_proj.*plugin_sw_unfused_test"):
        dispatch._refuse_unclassified(Model(Experts(plugin, affine, affine)))
    with pytest.raises(ValueError, match="up_proj.*plugin_sw_unfused_test"):
        dispatch._refuse_unclassified(Model(Experts(affine, plugin, affine)))
    with pytest.raises(ValueError, match="down_proj.*plugin_sw_unfused_test"):
        dispatch._refuse_unclassified(Model(Experts(affine, affine, plugin)))
    dispatch._refuse_unclassified(Model(Experts(affine, affine, affine)))


def test_refuse_unclassified_rejects_affine_subclass_plugin(tmp_path):
    import json

    import mlx.nn as nn

    from omlx.custom_kernels.quant_matmul import fast
    from omlx.formats import testing as fmt_testing
    from omlx.formats.affine import AffineFormat
    from omlx.formats.linear import FormatLinear, FormatSwitchLinear
    from omlx.patches import qwen35_oq_a8 as dispatch
    from omlx.patches.format_load import loading

    class Plugin(AffineFormat):
        mode = "custom_affine_test"

    class SameModeSubclass(AffineFormat):
        pass

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "omlx_quant_formats": {"affine": {}, Plugin.mode: {}},
            }
        )
    )

    stacked = mx.random.normal((2, 16, 256)).astype(mx.float16)
    parts = [mx.quantize(stacked[i], bits=4, group_size=64) for i in range(2)]
    sw_tensors = {
        "weight": mx.stack([part[0] for part in parts]),
        "scales": mx.stack([part[1] for part in parts]),
        "biases": mx.stack([part[2] for part in parts]),
    }
    dense_w = mx.random.normal((64, 256)).astype(mx.float16)
    dense_q4 = dict(
        zip(("weight", "scales", "biases"), mx.quantize(dense_w, bits=4, group_size=64))
    )
    dense_q6 = dict(
        zip(("weight", "scales", "biases"), mx.quantize(dense_w, bits=6, group_size=64))
    )
    with fmt_testing.verified("qwen4_exp"), loading(tmp_path):
        sw_plugin = FormatSwitchLinear(Plugin(), 4, 64, 256, sw_tensors)
        sw_affine = FormatSwitchLinear("affine", 4, 64, 256, sw_tensors)
        lin_plugin = FormatLinear(Plugin(), 4, 64, 256, dense_q4)
        lin_bad_mode = FormatLinear("affine", 4, 64, 256, dense_q4)
        lin_subclass_same_mode = FormatLinear("affine", 4, 64, 256, dense_q4)
        lin_stock_q6 = FormatLinear("affine", 6, 64, 256, dense_q6)
    object.__setattr__(lin_bad_mode, "mode", "custom_mode")
    object.__setattr__(lin_subclass_same_mode, "fmt", SameModeSubclass())

    class Experts(nn.Module):
        def __init__(self, gate_up, down):
            super().__init__()
            self.gate_up_proj = gate_up
            self.down_proj = down

    Experts.__module__ = "mlx_lm.models.switch_layers"

    class SwitchModel(nn.Module):
        def __init__(self, experts):
            super().__init__()
            self.switch_mlp = experts

    SwitchModel.__module__ = "mlx_vlm.models.qwen4_exp.language"

    with pytest.raises(ValueError, match="gate_up_proj.*custom_affine_test"):
        dispatch._refuse_unclassified(SwitchModel(Experts(sw_plugin, sw_affine)))

    class DenseBlock(nn.Module):
        def __init__(self, out_proj):
            super().__init__()
            self.q_proj = _quantized_linear(256, 128, 4)
            self.out_proj = out_proj

    DenseBlock.__module__ = "mlx_vlm.models.qwen4_exp.language"

    with pytest.raises(ValueError, match="out_proj.*custom_affine_test"):
        dispatch._refuse_unclassified(DenseBlock(lin_plugin))
    with pytest.raises(ValueError, match="out_proj.*custom_mode"):
        dispatch._refuse_unclassified(DenseBlock(lin_bad_mode))
    with pytest.raises(ValueError, match="out_proj.*affine"):
        dispatch._refuse_unclassified(DenseBlock(lin_subclass_same_mode))
    assert dispatch.classify_linear(lin_subclass_same_mode) is None
    # Genuine stock affine Q6 (unclassified for A8) is still accepted when
    # the native A8 is present; without it every projection classifies None
    # and the dead-opt-in guard is the correct answer.
    if fast.oq_a8_available():
        dispatch._refuse_unclassified(DenseBlock(lin_stock_q6))


def test_refuse_unclassified_unmeasured_qwen35_moe_switch_and_non_qwen(tmp_path):
    import json

    import mlx.nn as nn

    from omlx.formats import QuantFormat
    from omlx.formats import testing as fmt_testing
    from omlx.formats.linear import FormatSwitchLinear
    from omlx.patches import qwen35_oq_a8 as dispatch
    from omlx.patches.format_load import loading

    class Plugin(QuantFormat):
        mode = "plugin_q35_test"
        codes_name = "weight"
        tensors = ("scales", "biases")
        bits = (4,)
        group_sizes = (64,)

        def layout(self, k, n, bits, group_size):
            return {
                "weight": ((n, k * bits // 32), mx.uint32),
                "scales": ((n, k // group_size), None),
                "biases": ((n, k // group_size), None),
            }

        def decode(self, tensors, bits, group_size):
            return mx.dequantize(
                tensors["weight"],
                tensors["scales"],
                tensors["biases"],
                group_size=group_size,
                bits=bits,
                mode="affine",
            )

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "omlx_quant_formats": {"affine": {}, Plugin.mode: {}},
            }
        )
    )
    stacked = mx.random.normal((2, 16, 256)).astype(mx.float16)
    parts = [mx.quantize(stacked[i], bits=4, group_size=64) for i in range(2)]
    tensors = {
        "weight": mx.stack([part[0] for part in parts]),
        "scales": mx.stack([part[1] for part in parts]),
        "biases": mx.stack([part[2] for part in parts]),
    }
    with fmt_testing.verified("qwen4_exp"), loading(tmp_path):
        plugin = FormatSwitchLinear(Plugin(), 4, 64, 256, tensors)
        affine = FormatSwitchLinear("affine", 4, 64, 256, tensors)

    class Experts(nn.Module):
        def __init__(self, up, down):
            super().__init__()
            self.gate_up_proj = up
            self.down_proj = down

    Experts.__module__ = "mlx_lm.models.switch_layers"

    class SparseMoeBlock(nn.Module):
        def __init__(self, experts):
            super().__init__()
            self.switch_mlp = experts

    SparseMoeBlock.__module__ = "mlx_lm.models.qwen3_next"

    class DecoderLayer(nn.Module):
        def __init__(self, experts):
            super().__init__()
            self.mlp = SparseMoeBlock(experts)

    DecoderLayer.__module__ = "mlx_lm.models.qwen3_5"

    class Model(nn.Module):
        def __init__(self, experts):
            super().__init__()
            self.layer = DecoderLayer(experts)

    Model.__module__ = "mlx_lm.models.qwen3_5"

    model = Model(Experts(plugin, affine))
    assert dispatch._is_measured_family(model) is False
    with pytest.raises(ValueError, match="gate_up_proj.*plugin_q35_test"):
        dispatch._refuse_unclassified(model)

    # Reverse visit order: switch_layers visited before the mlx_lm.models.qwen3_5
    # module, proving the single pass preserves switch_err across iterations.
    class QwenMarker(nn.Module):
        pass

    QwenMarker.__module__ = "mlx_lm.models.qwen3_5"

    class Outer(nn.Module):
        def __init__(self, experts):
            super().__init__()
            self.first = SparseMoeBlock(experts)
            self.second = QwenMarker()

    Outer.__module__ = "mlx_lm.models.qwen3_next"
    experts_rev = Experts(plugin, affine)
    outer = Outer(experts_rev)
    visited = [m for _, m in outer.named_modules()]
    if visited.index(experts_rev) > visited.index(outer.second):
        outer = Outer.__new__(Outer)
        nn.Module.__init__(outer)
        outer.second = QwenMarker()
        outer.first = SparseMoeBlock(experts_rev)
        visited = [m for _, m in outer.named_modules()]
    assert visited.index(experts_rev) < visited.index(outer.second)
    with pytest.raises(ValueError, match="gate_up_proj.*plugin_q35_test"):
        dispatch._refuse_unclassified(outer)

    # Unmeasured mlx_vlm qwen3_5_moe model with switch_mlp.
    class VlmModel(nn.Module):
        def __init__(self, experts):
            super().__init__()
            self.switch_mlp = experts

    VlmModel.__module__ = "mlx_vlm.models.qwen3_5_moe.language"
    vlm_model = VlmModel(Experts(plugin, affine))
    assert dispatch._is_measured_family(vlm_model) is False
    with pytest.raises(ValueError, match="gate_up_proj.*plugin_q35_test"):
        dispatch._refuse_unclassified(vlm_model)

    # Non-Qwen model holding the same SwitchGLU is not refused.
    class LlamaModel(nn.Module):
        def __init__(self, experts):
            super().__init__()
            self.mlp = SparseMoeBlock(experts)

    LlamaModel.__module__ = "mlx_lm.models.llama"
    dispatch._refuse_unclassified(LlamaModel(Experts(plugin, affine)))

    # Unmeasured Qwen model whose plugin SwitchGLU lives in MTP is excluded.
    class MtpModel(nn.Module):
        def __init__(self, experts):
            super().__init__()
            self.layer = QwenMarker()
            self.mtp = SparseMoeBlock(experts)

        def get_mtp_module(self):
            return self.mtp

    MtpModel.__module__ = "mlx_lm.models.qwen3_5"
    dispatch._refuse_unclassified(MtpModel(Experts(plugin, affine)))
