# SPDX-License-Identifier: Apache-2.0
"""Integer levels, their affine map, and the sg8 traversal that consumes them."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx import formats
from omlx.formats import affine, testing
from omlx.formats.linear import FormatLinear
from omlx.patches import qwen35_verify_qmm as live
from tests.oracles import sg8 as oracle

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture(autouse=True)
def by_hand():
    with testing.verified("qwen4_exp"):
        yield


def _bits_equal(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert mx.array_equal(a.view(mx.uint8), b.view(mx.uint8)).item()


def _matrix(n, k, bits, group):
    # Independent rows, groups, and eight-code runs, including the last one.
    w = mx.random.normal((n, k), key=mx.random.key(2))
    return mx.quantize(w, group_size=group, bits=bits)


def _layer(n, k, bits, group, dtype):
    weight, scales, biases = _matrix(n, k, bits, group)
    return FormatLinear(
        "affine",
        bits,
        group,
        k,
        dict(weight=weight, scales=scales.astype(dtype), biases=biases.astype(dtype)),
    )


@pytest.mark.parametrize("bits", (2, 3, 4, 5, 6))
@pytest.mark.parametrize("group", (32, 64, 128))
def test_affine_codes_give_the_levels_and_the_weights(bits, group):
    k, n = 1024, 17
    layer = _layer(n, k, bits, group, mx.float16)
    face = formats.codes(layer, k, n)
    assert isinstance(face, formats.Codes)
    args = [layer[key] for key in ("weight", *face.tensors)]
    rows = [16, 0, 3, 16]
    levels, weights = testing.codes_rows(face, args, k, rows, mx.bfloat16)
    scales, biases = (a.astype(mx.float32) for a in args[1:])
    expected = mx.dequantize(args[0], scales, biases, group_size=group, bits=bits)
    codes = mx.dequantize(
        args[0],
        mx.ones_like(scales),
        mx.zeros_like(biases),
        group_size=group,
        bits=bits,
    )
    _bits_equal(weights, expected[rows])
    _bits_equal(levels, codes[rows].astype(mx.uint16))


@pytest.mark.parametrize("bits,group", [(8, 64), (4, 16)])
def test_affine_has_no_codes_face_outside_the_sg8_widths(bits, group):
    n, k = 16, 512
    layer = FormatLinear(
        "affine",
        bits,
        group,
        k,
        dict(
            weight=mx.zeros((n, k * bits // 32), mx.uint32),
            scales=mx.ones((n, k // group), mx.float16),
            biases=mx.zeros((n, k // group), mx.float16),
        ),
    )
    with testing.no_dot_defects():
        assert formats.codes(layer, k, n) is None


@pytest.mark.parametrize(
    "name,change",
    [
        ("weight", lambda a: a.astype(mx.int32)),
        ("weight", lambda a: a[:, :-1]),
        ("scales", lambda a: a[:, 1:]),
        ("biases", lambda a: a[:-1]),
        ("scales", lambda a: a.astype(mx.uint16)),
    ],
)
def test_affine_refuses_invalid_codes_storage(name, change):
    # Stock MLX layers accept the malformed tensors exercised here.
    layer = nn.QuantizedLinear(512, 16, bias=False, bits=4, group_size=64)
    assert isinstance(formats.codes(layer, 512, 16), formats.Codes)
    layer[name] = change(layer[name])
    with testing.no_dot_defects():
        assert formats.codes(layer, 512, 16) is None


# A format with byte-wide, complemented codes and interleaved scale/bias.
# Its matrix has one side tensor, and its raw bytes cannot feed affine sg8.
_BYTE_CODES = """
struct ByteCodes {
  static constant constexpr int CONTRACT = 1;
  const device uchar* values;
  const device float* maps;
  template <int K>
  struct Runs {
    static constant constexpr int GROUP = 64;
    const device uchar* values;
    const device float* maps;
    Runs(ByteCodes m, size_t row)
      : values(m.values + row * K), maps(m.maps + row * (K / GROUP) * 2) {}
    float2 params(int g) const { return float2(maps[2*g], maps[2*g+1]); }
    struct Pack { uchar4 lo, hi; };
    Pack fetch(int run) const {
      return ((const device Pack*)values)[run];
    }
    static uint pair(Pack q, int j) {
      return uint(q.lo[j] ^ 15) | (uint(q.hi[j] ^ 15) << 16);
    }
  };
};
"""


class ByteFormat(formats.QuantFormat):
    mode = "test_sg8_byte_codes"
    codes_name = "codes"
    tensors = ("maps",)
    contract = 1

    def codes(self, layer, k, n):
        return formats.Codes("byte_codes", _BYTE_CODES, "ByteCodes", self.tensors)


def _byte_layer(k, n, monkeypatch):
    fmt = ByteFormat()
    monkeypatch.setitem(formats._FORMATS, fmt.mode, fmt)
    packed, scales, biases = _matrix(n, k, 4, 64)
    codes = mx.dequantize(
        packed, mx.ones_like(scales), mx.zeros_like(biases), group_size=64, bits=4
    ).astype(mx.uint8)
    maps = mx.stack([scales, biases], axis=-1)
    layer = FormatLinear(fmt.mode, 4, 64, k, dict(codes=codes ^ 15, maps=maps))
    return layer, (packed, scales, biases)


@pytest.mark.parametrize("rows", (4, 5, 6, 7, 8))
@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16))
def test_single_sg8_reads_the_formats_own_codes(monkeypatch, rows, dtype):
    k, n = 512, 1024
    layer, affine = _byte_layer(k, n, monkeypatch)
    face = formats.codes(layer, k, n)
    x = mx.random.normal((rows, k), key=mx.random.key(4)).astype(dtype)
    for sums in (None, oracle.group_sums(x)):
        got = live.sg8_codes(
            x, face, (layer.codes, layer.maps), n=n, group_size=64, sums=sums
        )
        want = oracle.vk_qmm_sg8(x, *affine, group_size=64, sums=sums)
        _bits_equal(got, want)


def test_codes_kit_detects_levels_outside_the_sg8_operand(monkeypatch):
    k, n = 512, 16
    layer, _ = _byte_layer(k, n, monkeypatch)
    face = formats.codes(layer, k, n)
    storage = mx.full((n, k), 127 ^ 15, dtype=mx.uint8)
    levels, _ = testing.codes_rows(face, (storage, layer.maps), k, [0], mx.float16)
    assert (levels == 127).all().item()
    for col in (2, 6):
        broken = mx.array(storage)
        broken[2, 8 + col] = 128 ^ 15
        with pytest.raises(AssertionError, match="127"):
            testing.codes_rows(face, (broken, layer.maps), k, [2], mx.float16)


def test_sg8_kernels_with_one_key_and_different_sources_in_two_threads(monkeypatch):
    from dataclasses import replace
    from threading import Barrier

    k, n = 512, 1024
    layer, affine = _byte_layer(k, n, monkeypatch)
    face = formats.codes(layer, k, n)
    plain = replace(face, source=face.source.replace("^ 15", "^ 0"))
    x = mx.random.normal((5, k), key=mx.random.key(3)).astype(mx.bfloat16)
    want = oracle.vk_qmm_sg8(x, *affine, group_size=64)
    storage = layer.codes
    complemented = storage ^ 15
    mx.eval(want, x, storage, complemented, layer.maps)
    start = Barrier(2)

    def run(f, w):
        start.wait(timeout=10)
        got = live.sg8_codes(x, f, (w, layer.maps), n=n, group_size=64)
        _bits_equal(got, want)

    with ThreadPoolExecutor(2) as pool:
        futures = [
            pool.submit(run, f, w) for f, w in ((face, storage), (plain, complemented))
        ]
        for future in futures:
            future.result()


@pytest.mark.parametrize(
    "scale,bias,x_type",
    [
        (mx.bfloat16, mx.float32, mx.float16),
        (mx.float32, mx.float16, mx.bfloat16),
        (mx.float16, mx.bfloat16, mx.float16),
    ],
)
def test_sg8_reads_each_parameters_stored_type(scale, bias, x_type):
    k, n = 512, 1024
    # D11: a format layer takes a tensor of its own shape and dtype only while
    # a load replaces it, so the mixed stored dtypes are built in the layer.
    weight, scales, biases = _matrix(n, k, 5, 64)
    layer = FormatLinear(
        "affine",
        5,
        64,
        k,
        dict(
            weight=weight,
            scales=scales.astype(scale),
            biases=biases.astype(bias),
        ),
    )
    face = formats.codes(layer, k, n)
    operands = [layer[name] for name in ("weight", *face.tensors)]
    x = mx.random.normal((5, k), key=mx.random.key(9)).astype(x_type)
    _bits_equal(
        live.sg8_codes(x, face, operands, n=n, group_size=64),
        oracle.vk_qmm_sg8(x, *operands, bits=5, group_size=64),
    )


@pytest.mark.parametrize("detail", ["not a Codes", "no tensor"])
def test_codes_are_optional_and_use_the_registrys_operand_checks(monkeypatch, detail):
    k, n = 512, 16
    layer, _ = _byte_layer(k, n, monkeypatch)
    assert formats.QuantFormat().codes(layer, k, n) is None
    fmt = formats.get(layer.mode)
    if detail == "no tensor":
        result = formats.Codes("missing", "", "Codes", ("other",))
    else:
        result = formats.Tile("wrong_face", "", "Tile", ())
    monkeypatch.setattr(fmt, "codes", lambda *_: result)
    with testing.no_dot_defects() as defects:
        assert formats.codes(layer, k, n) is None
        assert len(defects) == 1
        mode, message = defects.pop()
        assert mode == layer.mode and detail in message


def test_affine_codes_read_flattened_rows_of_stacked_matrices():
    from omlx.formats.linear import FormatSwitchLinear

    n, k = 8, 256
    tensors = [
        mx.quantize(
            mx.random.normal((n, k), key=mx.random.key(i)).astype(mx.bfloat16),
            group_size=32,
            bits=3,
        )
        for i in (5, 6)
    ]
    w, s, b = (mx.stack(pair) for pair in zip(*tensors, strict=True))
    layer = FormatSwitchLinear("affine", 3, 32, k, dict(weight=w, scales=s, biases=b))
    face = formats.codes(layer, k, n)
    levels, weights = testing.codes_rows(face, (w, s, b), k, [15, 8, 7], mx.float16)
    float_s, float_b = s.astype(mx.float32), b.astype(mx.float32)
    want = mx.dequantize(w, float_s, float_b, group_size=32, bits=3)
    codes = mx.dequantize(
        w, mx.ones_like(float_s), mx.zeros_like(float_s), group_size=32, bits=3
    )
    _bits_equal(weights, want.reshape(2 * n, k)[[15, 8, 7]])
    _bits_equal(levels, codes.reshape(2 * n, k)[[15, 8, 7]].astype(mx.uint16))


@pytest.mark.parametrize(
    "k,rows,n,group", [(768, 5, 16, 64), (512, 9, 16, 64), (512, 5, 17, 64)]
)
def test_sg8_codes_rejects_incomplete_launch_geometry(monkeypatch, k, rows, n, group):
    # These three geometries would silently drop a K partition, row, or column.
    layer, _ = _byte_layer(k, n, monkeypatch)
    face = formats.codes(layer, k, n)
    operands = (layer.codes, layer.maps)
    with pytest.raises((ValueError, RuntimeError), match="sg8.*geometry"):
        mx.eval(
            live.sg8_codes(
                mx.ones((rows, k), mx.float16), face, operands, n=n, group_size=group
            )
        )


@pytest.mark.parametrize(
    "old,new,error",
    [
        ("CONTRACT = 1", "CONTRACT = 0", "unsupported contract revision"),
        (
            "CONTRACT = 1",
            f"CONTRACT = {formats.CONTRACT + 1}",
            "unsupported contract revision",
        ),
        ("GROUP = 64", "GROUP = 48", "whole groups"),
        ("GROUP = 64", "GROUP = 4", "whole groups"),
        ("float2 params", "half2 params", "params.*float2"),
        ("static uint pair", "static int pair", "pair.*uint"),
    ],
    ids=["old", "new", "partial-group", "partial-run", "params", "pair"],
)
def test_codes_struct_contract_is_checked(old, new, error):
    from dataclasses import replace

    face = formats.Codes("invalid_codes", _BYTE_CODES, "ByteCodes", ("maps",))
    source = face.source.replace(old, new)
    if old == "float2 params":
        source = source.replace("return float2(maps", "return half2(maps")
    # Overallocate the map even for the deliberately too-small group.
    operands = (mx.ones((1, 512), mx.uint8), mx.ones((1, 256), mx.float32))
    with pytest.raises((ValueError, RuntimeError), match=error):
        testing.codes_rows(replace(face, source=source), operands, 512, [0], mx.float16)


def test_sg8_reads_provided_group_sums(monkeypatch):
    k, n = 512, 1024
    layer, affine = _byte_layer(k, n, monkeypatch)
    face = formats.codes(layer, k, n)
    x = mx.random.normal((5, k), key=mx.random.key(14)).astype(mx.bfloat16)
    # An offset distinguishes reading the supplied sums from recomputing them.
    sums = oracle.group_sums(x) + 1.0
    want = oracle.vk_qmm_sg8(x, *affine, group_size=64, sums=sums)
    plain = oracle.vk_qmm_sg8(x, *affine, group_size=64)
    assert not mx.array_equal(want, plain).item()
    got = live.sg8_codes(
        x, face, (layer.codes, layer.maps), n=n, group_size=64, sums=sums
    )
    _bits_equal(got, want)


def test_sg8_matches_the_callers_group_size_to_the_face(monkeypatch):
    k, n = 512, 16
    layer, _ = _byte_layer(k, n, monkeypatch)
    face = formats.codes(layer, k, n)
    # The extra maps keep the invalid geometry inside the buffers in any case.
    operands = (layer.codes, mx.tile(layer.maps, (1, 2, 1)))
    with pytest.raises((ValueError, RuntimeError), match="codes face: group size"):
        mx.eval(
            live.sg8_codes(
                mx.ones((5, k), mx.float16), face, operands, n=n, group_size=32
            )
        )


@pytest.mark.parametrize(
    "field", ["weight", "bias dtype", "map width", "stack rank", "tiny"]
)
def test_affine_declines_incompatible_codes_without_reporting_a_defect(field):
    n, k = 8, 256
    layer = nn.QuantizedLinear(k, n, bias=False, bits=4, group_size=32)
    assert isinstance(formats.codes(layer, k, n), formats.Codes)
    if field == "weight":
        del layer["weight"]
    elif field == "bias dtype":
        layer.biases = layer.biases.astype(mx.uint16)
    elif field == "map width":
        layer.scales = layer.scales[:, :-1]
        layer.biases = layer.biases[:, :-1]
    elif field == "stack rank":
        for key in ("weight", "scales", "biases"):
            layer[key] = layer[key][None, None]
    else:
        n, k = 1, 32
        layer = _layer(n, k, 4, 32, mx.float16)
    with testing.no_dot_defects():
        assert formats.codes(layer, k, n) is None


def test_codes_rows_requires_a_codes_face():
    face = formats.Tile("tile", "", "Tile", ())
    with pytest.raises(ValueError, match="not a Codes face"):
        testing.codes_rows(face, (), 512, [0], mx.float16)


def test_codes_defect_report_names_the_method_or_the_face(monkeypatch):
    def refused(*args):
        raise ValueError("unusable storage")

    monkeypatch.setattr(ByteFormat, "codes", refused)
    layer, _ = _byte_layer(512, 16, monkeypatch)
    with pytest.raises(AssertionError) as raised, testing.no_dot_defects():
        assert formats.codes(layer, 512, 16) is None
    summary, detail = str(raised.value).split(": ", 1)
    assert "codes()" in summary or "face" in summary
    assert "unusable storage" in detail


_TABLE_CODES = """
struct TableCodes {
  static constant constexpr int CONTRACT = 1;
  const device uchar* values;
  const device float* maps;
  omlx_table<uint8_t, 16> lut;
  template <int K>
  struct Runs {
    static constant constexpr int GROUP = 64;
    const device uchar* values;
    const device float* maps;
    omlx_table<uint8_t, 16> lut;
    Runs(TableCodes m, size_t row)
        : values(m.values + row * K), maps(m.maps + row * (K / GROUP) * 2),
          lut(m.lut) {}
    float2 params(int g) const { return float2(maps[2*g], maps[2*g+1]); }
    struct Pack { uchar lo[4], hi[4]; };
    Pack fetch(int run) const {
      Pack p;
      for (int j = 0; j < 4; ++j) {
        p.lo[j] = lut[values[8*run+j]];
        p.hi[j] = lut[values[8*run+j+4]];
      }
      return p;
    }
    static uint pair(Pack p, int j) {
      return uint(p.lo[j]) | (uint(p.hi[j]) << 16);
    }
  };
};
"""


def test_staged_tables_reads_a_codes_faces_packed_table():
    bits = tuple(i ^ 11 for i in range(16))
    table = formats.Table("lut", mx.uint8, bits)
    face = formats.Codes(
        "table_codes", _TABLE_CODES, "TableCodes", ("maps",), tables=(table,)
    )
    operands = (mx.zeros((1, 512), mx.uint8), mx.ones((1, 8, 2), mx.float32))
    got = testing.staged_tables(face, operands, mx.float16)
    assert tuple(got) == ("lut",)
    _bits_equal(got["lut"], mx.array(bits, dtype=mx.uint8))


@pytest.mark.parametrize("consumer", ["kit", "single"])
def test_codes_read_their_own_staged_table(monkeypatch, consumer):
    k, n = 512, 1024
    x = mx.random.normal((5, k), key=mx.random.key(17)).astype(mx.bfloat16)
    rows = [7, 0, n - 1]

    # Same source and key. Only the bits staged for this table change.
    for mask in (15, 0):
        table = formats.Table("lut", mx.uint8, tuple(i ^ mask for i in range(16)))
        face = formats.Codes(
            "table_codes", _TABLE_CODES, "TableCodes", ("maps",), tables=(table,)
        )
        monkeypatch.setattr(ByteFormat, "codes", lambda *_, result=face: result)
        layer, (packed, scales, biases) = _byte_layer(k, n, monkeypatch)
        selected = formats.codes(layer, k, n)
        assert selected is face
        operands = (layer.codes, layer.maps)
        # _byte_layer stores q ^ 15. An identity table reads this complement
        # literally, equivalent to flipping each nibble of the oracle's words.
        words = packed if mask == 15 else ~packed
        if consumer == "kit":
            levels, weights = testing.codes_rows(
                selected, operands, k, rows, mx.float16
            )
            _bits_equal(levels, (layer.codes[rows] ^ mask).astype(mx.uint16))
            want = mx.dequantize(words, scales, biases, group_size=64, bits=4)
            _bits_equal(weights, want[rows])
        else:
            result = live.sg8_codes(x, selected, operands, n=n, group_size=64)
            want = oracle.vk_qmm_sg8(x, words, scales, biases, group_size=64)
            _bits_equal(result, want)


@pytest.mark.parametrize("shared", [False, True])
def test_codes_keeps_the_matrices_other_skeletons(shared):
    from omlx.formats import kernels
    from tests.test_formats_kernels import _toy

    matrices = kernels.Matrices()
    matrices.add("dot", _toy("beside_codes"), shared=shared)
    assert "struct omlx_code_runs" not in matrices.header
    face = formats.Codes("byte_codes", _BYTE_CODES, "ByteCodes", ("maps",))
    with pytest.raises(TypeError, match="a codes face has no shared source"):
        matrices.add("codes", face, shared=True)
    matrices.add("codes", face)
    header = matrices.header
    assert header.count("struct omlx_code_runs") == 1
    assert ("void qmv_rows(" in header) == (not shared)
    assert ("void qmv_shared(" in header) == shared


def test_codes_lookup_uses_each_layers_bound_format(monkeypatch):
    k, n = 512, 16
    first, _ = _byte_layer(k, n, monkeypatch)
    replacement = ByteFormat()
    monkeypatch.setattr(replacement, "codes", lambda *args: None)
    monkeypatch.setitem(formats._FORMATS, first.mode, replacement)
    second = FormatLinear(
        first.mode, 4, 64, k, dict(codes=first.codes, maps=first.maps)
    )
    assert formats.codes(second, k, n) is None
    face = formats.codes(first, k, n)
    assert isinstance(face, formats.Codes) and face.key == "byte_codes"


def test_sg8_codes_refuses_invalid_face_or_dtype(monkeypatch):
    """A tile face, a float32 activation, or a face exact for another dtype
    is refused by the entry, not left to Metal."""
    k, n = 512, 16
    layer, _ = _byte_layer(k, n, monkeypatch)
    face = formats.codes(layer, k, n)
    operands = (layer.codes, layer.maps)
    with pytest.raises(ValueError, match="not a Codes face"):
        mx.eval(
            live.sg8_codes(
                mx.ones((4, k), mx.bfloat16),
                affine.tile_face(4, 64, mx.float16),
                operands,
                n=n,
                group_size=64,
            )
        )
    with pytest.raises(ValueError, match="bf16 or fp16"):
        mx.eval(
            live.sg8_codes(
                mx.ones((4, k), mx.float32), face, operands, n=n, group_size=64
            )
        )
    only_bf16 = replace(face, dtypes=(mx.bfloat16,))
    with pytest.raises(ValueError, match="exact for"):
        mx.eval(
            live.sg8_codes(
                mx.ones((4, k), mx.float16),
                only_bf16,
                operands,
                n=n,
                group_size=64,
            )
        )


def test_affine_codes_face_refuses_bad_dtypes():
    with pytest.raises(ValueError, match="affine codes face"):
        affine.codes_face(4, 64, mx.uint16, mx.float16)
    with pytest.raises(ValueError, match="affine codes face"):
        affine.codes_face(4, 64, mx.float16, mx.uint16)


def test_affine_codes_face_lists_its_exact_dtypes():
    face = affine.codes_face(4, 64, mx.bfloat16, mx.float16)
    assert face.dtypes == (mx.bfloat16, mx.float16)
    assert face.exact_for(mx.bfloat16) and face.exact_for(mx.float16)
    assert not face.exact_for(mx.float32)


def test_affine_codes_face_refuses_unsupported_widths():
    with pytest.raises(ValueError, match="affine codes face"):
        affine.codes_face(7, 64, mx.bfloat16, mx.float16)
    with pytest.raises(ValueError, match="affine codes face"):
        affine.codes_face(4, 8, mx.bfloat16, mx.float16)


def test_codes_rows_names_the_codes_tensor_in_an_error():
    """``codes_name`` is the packed tensor's name in the check's message: a
    caller that names it differently sees its own name, not ``weight``."""
    face = formats.Codes("byte_codes", _BYTE_CODES, "ByteCodes", ("maps",))
    with pytest.raises(ValueError, match="codes is not an mx.array"):
        testing.codes_rows(
            face,
            (None, mx.ones((1, 8, 2), mx.float32)),
            512,
            [0],
            mx.float16,
            codes_name="codes",
        )


def test_a_codes_face_refuses_a_hand_written_table():
    """R2's rule reaches the codes kind: a table is declared as data, never
    written in the face's source."""
    source = _BYTE_CODES.replace(
        "struct ByteCodes {", "constant float LUT[4] = {1};\nstruct ByteCodes {"
    )
    with pytest.raises(ValueError, match="its source declares LUT"):
        formats.Codes("hand_table", source, "ByteCodes", ("maps",))


def test_sg8_codes_refuses_non_multiple_of_32_group():
    """A group of 40 has whole eight-code runs but not whole 32-wide blocks:
    the kernel reads 8 of every 40 weights and would drop the rest."""
    k, n = 320, 16
    face = formats.Codes(
        "forty", _BYTE_CODES.replace("GROUP = 64", "GROUP = 40"), "ByteCodes", ("maps",)
    )
    operands = (mx.zeros((n, k), mx.uint8), mx.ones((n, k // 40, 2), mx.float32))
    with pytest.raises((ValueError, RuntimeError), match="sg8.*geometry"):
        mx.eval(
            live.sg8_codes(
                mx.ones((5, k), mx.float16), face, operands, n=n, group_size=40
            )
        )


def _stock_pair(n, k, bits, group, scale, bias, seed):
    raw = mx.random.normal((n, k), key=mx.random.key(seed)) * 0.05
    weight, scales, biases = mx.quantize(raw, group_size=group, bits=bits)
    layer = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
    nn.Module.__init__(layer)
    layer.group_size, layer.bits, layer.mode = group, bits, "affine"
    layer.weight = weight
    layer.scales, layer.biases = scales.astype(scale), biases.astype(bias)
    layer.freeze()
    return layer


def test_swiglu_reads_each_matrixs_stored_types():
    k, n = 512, 1024
    gate = _stock_pair(n, k, 5, 64, mx.float16, mx.float32, 21)
    up = _stock_pair(n, k, 5, 64, mx.float32, mx.bfloat16, 22)
    x = mx.random.normal((4, k), key=mx.random.key(23)).astype(mx.bfloat16)
    _bits_equal(
        live.vk_swiglu_sg8(x, gate, up),
        oracle.vk_swiglu_sg8(x, gate, up),
    )


def test_affine_declines_non_whole_groups():
    layer = FormatLinear(
        "affine",
        4,
        64,
        544,
        dict(
            weight=mx.zeros((8, 68), mx.uint32),
            scales=mx.ones((8, 8), mx.float16),
            biases=mx.zeros((8, 8), mx.float16),
        ),
    )
    with testing.no_dot_defects():
        assert formats.codes(layer, 544, 8) is None
