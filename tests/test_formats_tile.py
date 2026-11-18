# SPDX-License-Identifier: Apache-2.0
"""The tile face: a format's second face of a matrix, read in runs of 8 codes
and given weight by weight. What a kernel checks of a face when it is
compiled, the kit's check (``testing.tile_rows``) with the page's example,
and the built-in affine tile face against MLX's dequantize in float32.
"""

from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import takewhile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx import formats
from omlx.formats import Dot, affine, kernels, testing

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

_PAGE = (Path(__file__).parents[1] / "docs" / "quant-formats.md").read_text()

# The page's example (docs/quant-formats.md, The tile face): 8-bit codes
# around 128, a float32 scale for each group of 32 codes of a row and one for
# the row, w = (code - 128) * (group_scale * row_scale).
_TOY = r"""namespace omlx_toy_KEY {
struct Tile {
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* codes;
  const device float* group_scales;
  const device float* row_scales;

  template <int K>
  struct Runs {
    static constant constexpr int RUN = 8;
    static constant constexpr int GROUP = 32;
    static constant constexpr int GROUPS = (K - 1) / GROUP + 1;
    struct P {
      float s;  // the group's scale times the row's
    };
    using Codes = uint2;  // 8 codes of 8 bits

    const device uint2* c;
    const device float* g;
    const device float* r;

    Runs(Tile m, size_t row)
        : c((const device uint2*)(m.codes + row * K)),
          g(m.group_scales + row * GROUPS),
          r(m.row_scales + row) {}

    P params(int group) const {
      return {g[group] * r[0]};
    }
    Codes fetch(int run) const {
      return c[run];
    }
    static float weight(thread const P& p, Codes q, int i) {
      const uint code = ((i < 4 ? q.x : q.y) >> (8 * (i & 3))) & 0xFFu;
      return (float(code) - 128.0f) * p.s;
    }
  };
};
}  // namespace omlx_toy_KEY
"""


def _toy(key, source=_TOY) -> formats.Tile:
    return formats.Tile(
        key=f"toy_{key}",
        source=source.replace("KEY", key),
        type=f"omlx_toy_{key}::Tile",
        tensors=("group_scales", "row_scales"),
    )


def _toy_matrix(rows, k, seed=0, group=32):
    """The toy's tensors for ``rows`` rows of K inputs: random codes, and
    scales with every bit of a float32 mantissa, so that the order of the
    two products shows."""
    mx.random.seed(seed)
    codes = mx.random.randint(0, 256, (rows, k)).astype(mx.uint8)
    group_scales = mx.random.normal((rows, (k + group - 1) // group))
    row_scales = mx.random.normal((rows,))
    return codes, group_scales, row_scales


def _toy_decode(codes, group_scales, row_scales, group=32):
    """The toy's decode in float32, as the page gives it."""
    levels = codes.astype(mx.float32) - 128.0
    scales = group_scales * row_scales[..., None]
    return levels * scales[..., mx.arange(levels.shape[-1]) // group]


def _same(a, b) -> bool:
    """Bit for bit, and a NaN as a NaN whatever its bits."""
    assert a.dtype == b.dtype and a.shape == b.shape
    bits = {4: mx.uint32, 2: mx.uint16}[a.dtype.size]
    nan = mx.isnan(a)
    return (
        mx.array_equal(nan, mx.isnan(b)).item()
        and mx.array_equal(
            mx.where(nan, 0, a.view(bits)), mx.where(nan, 0, b.view(bits))
        ).item()
    )


def _error(face, operands, k=64) -> str:
    with pytest.raises(RuntimeError) as failure:
        mx.eval(testing.tile_rows(face, operands, k, range(8), mx.float32))
    return str(failure.value)


# --- The face in Python ---------------------------------------------------


def test_a_tile_is_a_face_of_its_own_kind():
    tile = _toy("kind")
    dot = Dot(**{f: getattr(tile, f) for f in ("key", "source", "type", "tensors")})
    assert (
        tile != dot and not isinstance(tile, Dot) and not isinstance(dot, formats.Tile)
    )
    assert len({tile, dot}) == 2
    # Equal tiles are one cache key, as equal dots are.
    assert replace(tile) == tile and hash(replace(tile)) == hash(tile)
    assert type(replace(tile)) is formats.Tile


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"key": "toy-tile"}, "is not an identifier"),
        ({"tensors": ("row scales",)}, "is not an identifier"),
        ({"tensors": ["row_scales"]}, "tensors and dtypes are tuples"),
        (
            {"tensors": ("row_scales", "row_scales")},
            "beside 'weight' and 'bias', each named",
        ),
        ({"tensors": ("weight",)}, "beside 'weight' and 'bias', each named once"),
        ({"tensors": ("bias",)}, "beside 'weight' and 'bias', each named once"),
        ({"dtypes": (mx.float16, mx.float16)}, "dtypes are distinct mx.Dtype"),
    ],
)
def test_a_tile_is_checked_as_a_dot_is(fields, message):
    with pytest.raises(ValueError, match=message):
        replace(_toy("checked"), **fields)


def test_a_format_has_no_tile_face_until_it_gives_one():
    assert formats.QuantFormat().tile(nn.Module(), 64, 8) is None


# --- What a kernel includes -----------------------------------------------


def _tile_line() -> int:
    """The line of the skeletons that the tile skeleton starts on: after the
    index packs, the revisions and the row skeleton."""
    return (
        kernels._INDEX_PACKS.count("\n")
        + 1
        + (kernels._revisions() + kernels._QMV_ROWS).count("\n")
    )


def _header(*sources, rows=False) -> str:
    """The header of a kernel of faces of these ``(key, source)``, with the
    row skeleton if ``rows``, and the tile skeleton: each skeleton at its own
    lines of the skeletons."""
    named = kernels._named
    return (
        named(kernels._SKELETONS, kernels._INDEX_PACKS)
        + "".join(named(f"<quant format {key}>", source) for key, source in sources)
        + named(
            kernels._SKELETONS,
            kernels._revisions() + (kernels._QMV_ROWS if rows else ""),
            line=kernels._INDEX_PACKS.count("\n") + 1,
        )
        + named(kernels._SKELETONS, kernels._TILE_FACE, line=_tile_line())
        + kernels._REVISIONS_END
        + named("<kernel header>", "")
    )


def test_a_kernel_of_tile_faces_has_the_tile_skeleton_alone():
    matrices = kernels.Matrices()
    face = _toy("header")
    assert matrices.add("m", face) == ["m_weight", "m_group_scales", "m_row_scales"]
    assert matrices.header == _header((face.key, face.source))
    assert matrices.declarations.startswith(
        "    const auto m = omlx_toy_header::Tile{(const device uint8_t*)m_weight, "
        "m_group_scales, m_row_scales};\n"
    )


def test_a_kernel_of_both_kinds_has_both_skeletons():
    dot, tile = affine.face(4, 64, True, mx.bfloat16), _toy("both")
    matrices = kernels.Matrices()
    matrices.add("d", dot)
    matrices.add("t", tile)
    matrices.add("u", replace(tile, key="toy_both_again"))
    assert matrices.header == _header(
        (dot.key, dot.source), (tile.key, tile.source), rows=True
    )


def test_a_kernel_of_no_matrix_ends_the_macros_on_a_line_of_their_own():
    header = kernels.Matrices().header
    assert (
        f"#define OMLX_OLDEST_CONTRACT {formats.OLDEST_CONTRACT}\n"
        "#undef OMLX_CONTRACT\n#undef OMLX_OLDEST_CONTRACT\n"
    ) in header
    assert "omlx_tile" not in header and "qmv_rows" not in header


def test_a_matrix_is_read_through_a_dot_or_a_tile():
    matrices = kernels.Matrices()
    for face in (object(), "toy", None):
        with pytest.raises(TypeError, match="is not a Dot or a Tile"):
            matrices.add("m", face)
    assert matrices.add("m", _toy("added")) == [
        "m_weight",
        "m_group_scales",
        "m_row_scales",
    ]


# The sha256 of the header and the declarations of a kernel of one affine dot
# face, as oMLX's dot kernels have them at 059e4a8: a dot kernel's source is
# the parent's to the byte, so that every kernel a checkpoint runs is too.
_DOT_KERNEL = "8247e35050747bd5c6b15961714316d1a1eb7aa8c2bf40fbea127643fe7079aa"


def test_a_kernel_of_dot_faces_is_written_as_before_tile_faces():
    matrices = kernels.Matrices()
    matrices.add("m", affine.face(4, 64, True, mx.bfloat16))
    text = matrices.header + kernels.STORE_ROWS + matrices.declarations
    assert "omlx_tile" not in text
    assert hashlib.sha256(text.encode()).hexdigest() == _DOT_KERNEL


# --- What a kernel checks of a face ----------------------------------------


def _toy_operands(k=64, rows=8):
    return _toy_matrix(rows, k)


def _both_error(face, operands, k=64) -> str:
    """The compiler's messages for ``face`` in a kernel that reads a dot face
    too: the tile skeleton follows the row skeleton there."""
    matrices = kernels.Matrices()
    inputs = ["x", *matrices.add("d", affine.face(4, 64, True, mx.bfloat16))]
    inputs += matrices.add("t", face)
    kernel = mx.fast.metal_kernel(
        name=f"omlx_test_both_kinds_{face.key}",
        input_names=inputs,
        output_names=["w32", "wt"],
        header=matrices.header + testing._TILE_ROWS_HEADER,
        source=matrices.declarations + r"""
    const uint w = thread_position_in_grid.x;
    omlx_tile_rows<T, K>(t, 0, int(w), w32 + w, wt + w);
""",
    )
    layer = _layer(512, 8, 4, 64)
    with pytest.raises(RuntimeError) as failure:
        mx.eval(
            kernel(
                inputs=[
                    mx.zeros((512,), mx.bfloat16),
                    layer.weight,
                    layer.scales,
                    layer.biases,
                    *operands,
                ],
                template=[("T", mx.bfloat16), ("K", k)],
                grid=(k, 1, 1),
                threadgroup=(k, 1, 1),
                output_shapes=[(k,), (k,)],
                output_dtypes=[mx.float32, mx.bfloat16],
            )
        )
    return str(failure.value)


# A rule each face breaks, the edit of the toy that breaks it (or the edits),
# and its sentence. The compiler cites the sentence at the skeleton's line
# that states it, and says nothing else.
_BREAKS = {
    "contract": (
        ("  static constant constexpr int CONTRACT = 1;\n", ""),
        "tile face: the face needs a static constant constexpr int CONTRACT: the "
        "revision of the contract it is written for",
    ),
    "contract_a_float": (
        ("int CONTRACT = 1;", "float CONTRACT = 1.0f;"),
        "tile face: the face needs a static constant constexpr int CONTRACT",
    ),
    "runs_misnamed": (
        [
            ("  struct Runs {", "  struct Cursor {"),
            ("    Runs(Tile", "    Cursor(Tile"),
        ],
        "tile face: the face needs a member template Runs<int K>: the cursor of a "
        "row of K inputs",
    ),
    "run": (
        ("RUN = 8;", "RUN = 16;"),
        "tile face: Runs<K>::RUN must be 8: every tile kernel reads runs of 8 codes",
    ),
    "run_misnamed": (("int RUN = 8;", "int RUNS = 8;"), "Runs<K>::RUN must be 8"),
    "run_not_static": (
        ("    static constant constexpr int RUN = 8;", "    int RUN = 8;"),
        "Runs<K>::RUN must be 8",
    ),
    "group_of_12": (
        ("GROUP = 32;", "GROUP = 12;"),
        "tile face: Runs<K>::GROUP must be 8 times a power of two (8, 16, 32, ..., "
        "2^30): the values a kernel's thread decodes are then whole groups or in one",
    ),
    "group_of_24": (("GROUP = 32;", "GROUP = 24;"), "Runs<K>::GROUP must be 8 times"),
    "group_of_4": (("GROUP = 32;", "GROUP = 4;"), "Runs<K>::GROUP must be 8 times"),
    "group_past_an_int": (
        ("int GROUP = 32;", "size_t GROUP = size_t(1) << 32;"),
        "Runs<K>::GROUP must be 8 times",
    ),
    "group_a_float": (
        ("int GROUP = 32;", "float GROUP = 32.0f;"),
        "Runs<K>::GROUP must be 8 times",
    ),
    "group_misnamed": (
        [
            ("int GROUP = 32;", "int GROUPED = 32;"),
            ("(K - 1) / GROUP + 1;", "(K - 1) / GROUPED + 1;"),
        ],
        "Runs<K>::GROUP must be 8 times",
    ),
    "held": (
        (
            "    struct P {\n      float s;",
            "    struct P {\n      P(float v) : s(v) {}\n      float s;",
        ),
        "tile face: Runs<K>::P and Runs<K>::Codes must be types that are "
        "default-constructible and copy-assignable: a kernel holds arrays of them",
    ),
    "held_not_assignable": (
        ("      float s;  //", "      const float s = 0.0f;  //"),
        "Runs<K>::P and Runs<K>::Codes must be types",
    ),
    "codes_a_reference": (
        (
            "using Codes = uint2;  // 8 codes of 8 bits",
            "using Codes = const device uint2&;  // 8 codes of 8 bits",
        ),
        "Runs<K>::P and Runs<K>::Codes must be types",
    ),
    "p_misnamed": (
        [
            ("    struct P {", "    struct Q {"),
            ("    P params(", "    Q params("),
            ("thread const P& p", "thread const Q& p"),
        ],
        "Runs<K>::P and Runs<K>::Codes must be types",
    ),
    "codes_misnamed": (
        [
            ("using Codes = uint2;", "using Code = uint2;"),
            ("    Codes fetch(", "    Code fetch("),
            ("Codes q, int i", "Code q, int i"),
        ],
        "Runs<K>::P and Runs<K>::Codes must be types",
    ),
    "cursor": (
        ("Runs(Tile m, size_t row)\n", "Runs(Tile m, size_t row, int k)\n"),
        "tile face: Runs<K> must be constructible from the face and a size_t row",
    ),
    "cursor_of_a_moved_face": (
        ("Runs(Tile m, size_t row)\n", "Runs(thread Tile&& m, size_t row)\n"),
        "Runs<K> must be constructible from the face and a size_t row",
    ),
    "params_not_const": (
        ("P params(int group) const {", "P params(int group) {"),
        "tile face: Runs<K> needs a const member params(int g) that returns a P",
    ),
    "params_misnamed": (
        ("P params(int group) const {", "P param(int group) const {"),
        "Runs<K> needs a const member params(int g)",
    ),
    "params_of_another_type": (
        (
            "    P params(int group) const {\n      return {g[group] * r[0]};\n",
            "    float params(int group) const {\n      return g[group] * r[0];\n",
        ),
        "Runs<K> needs a const member params(int g)",
    ),
    "params_of_a_pointer": (
        (
            "    P params(int group) const {\n      return {g[group] * r[0]};\n",
            "    P params(const device int*) const {\n      return {g[0] * r[0]};\n",
        ),
        "Runs<K> needs a const member params(int g)",
    ),
    "fetch_misnamed": (
        ("Codes fetch(int run) const {", "Codes fetches(int run) const {"),
        "tile face: Runs<K> needs a const member fetch(int run) that returns a Codes",
    ),
    "fetch_not_const": (
        ("Codes fetch(int run) const {", "Codes fetch(int run) {"),
        "Runs<K> needs a const member fetch(int run)",
    ),
    "fetch_of_another_type": (
        (
            "    Codes fetch(int run) const {\n      return c[run];\n",
            "    uint fetch(int run) const {\n      return c[run].x;\n",
        ),
        "Runs<K> needs a const member fetch(int run)",
    ),
    "weight_not_static": (
        ("static float weight(", "float weight("),
        "tile face: Runs<K> needs a static member weight(P, Codes, int i) that "
        "returns a float",
    ),
    "weight_a_const_member": (
        (
            "static float weight(thread const P& p, Codes q, int i) {",
            "float weight(thread const P& p, Codes q, int i) const {",
        ),
        "Runs<K> needs a static member weight(P, Codes, int i)",
    ),
    "weight_of_a_moved_run": (
        ("Codes q, int i) {", "thread Codes&& q, int i) {"),
        "Runs<K> needs a static member weight(P, Codes, int i)",
    ),
    "weight_of_another_type": (
        ("static float weight(", "static half weight("),
        "Runs<K> needs a static member weight(P, Codes, int i)",
    ),
}


def _broken(rule) -> str:
    """The toy's source with ``rule`` broken."""
    edits, _ = _BREAKS[rule]
    source = _TOY
    for old, new in edits if isinstance(edits, list) else [edits]:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source


@metal
@pytest.mark.parametrize("rule", _BREAKS)
def test_a_tile_face_that_breaks_a_rule_is_told_that_rule(rule):
    _, sentence = _BREAKS[rule]
    message = _error(_toy(f"break_{rule}", _broken(rule)), _toy_operands())
    assert sentence in message
    # The rule, and no error of the kernel's that uses the face after it.
    assert message.count("error:") == 1
    # Cited at the line of the skeletons that states it.
    (line,) = re.findall(
        r"<omlx skeletons, docs/quant-formats\.md>:(\d+):\d+: error", message
    )
    text = kernels._TILE_FACE.split("\n")[int(line) - _tile_line()]
    assert text.strip() == "static_assert("


@metal
def test_a_rule_is_cited_at_its_line_whatever_else_the_kernel_reads():
    _, sentence = _BREAKS["run"]
    face = _toy("cited", _broken("run"))
    alone = _error(face, _toy_operands())
    beside_a_dot = _both_error(replace(face, key="toy_cited_beside"), _toy_operands())
    cited = r"<omlx skeletons, docs/quant-formats\.md>:\d+:\d+: error"
    assert re.findall(cited, alone) == re.findall(cited, beside_a_dot) != []
    assert sentence in beside_a_dot and beside_a_dot.count("error:") == 1


@metal
@pytest.mark.parametrize(
    "contract", [formats.CONTRACT + 1, formats.OLDEST_CONTRACT - 1]
)
def test_a_tile_face_of_another_revision_is_told_that_and_nothing_else(contract):
    # As this oMLX may meet one: breaking rules, without a member it calls.
    source = (
        _TOY.replace("CONTRACT = 1;", f"CONTRACT = {contract};")
        .replace("RUN = 8;", "RUN = 16;")
        .replace("Codes fetch(int run) const {", "Codes fetches(int run) const {")
    )
    message = _error(_toy(f"another_{contract}", source), _toy_operands())
    assert "tile face: written for a" in message
    assert message.count("error:") == 1


def _revision(contract) -> formats.Tile:
    return _toy(
        f"revision_{contract}", _TOY.replace("CONTRACT = 1;", f"CONTRACT = {contract};")
    )


@metal
def test_a_kernel_runs_the_tile_faces_of_the_revisions_omlx_runs(monkeypatch):
    # As an oMLX of revision 3 that still runs the formats of revision 2.
    monkeypatch.setattr(formats, "CONTRACT", 3)
    monkeypatch.setattr(formats, "OLDEST_CONTRACT", 2)
    operands = _toy_operands()
    want = _toy_decode(*operands)
    for contract in (2, 3):
        exact, _ = testing.tile_rows(
            _revision(contract), operands, 64, range(8), mx.float32
        )
        assert _same(exact, want)
    for contract, rule, revisions in (
        (4, "a later revision of the contract than this oMLX runs", "'4 <= 3'"),
        (1, "a revision of the contract this oMLX no longer runs", "'1 >= 2'"),
    ):
        message = _error(_revision(contract), operands)
        assert f"tile face: written for {rule}" in message
        # The face's revision against oMLX's, in numbers.
        assert f"due to requirement {revisions}" in message
        assert "<omlx skeletons, docs/quant-formats.md>:" in message
        assert message.count("tile face: ") == 1 and "<quant format" not in message


@metal
def test_a_tile_face_that_breaks_several_rules_is_told_the_first():
    source = _TOY
    for rule in ("params_not_const", "group_of_12", "run"):
        (old, new), _ = _BREAKS[rule]
        source = source.replace(old, new)
    message = _error(_toy("several", source), _toy_operands())
    assert _BREAKS["run"][1] in message and message.count("error:") == 1


# Groups of 8, and of 2^30: the smallest and the largest of the rule.
@metal
@pytest.mark.parametrize("group", [8, 1 << 30])
def test_a_tile_face_of_the_smallest_and_the_largest_groups_runs(group):
    face = _toy(f"group_{group}", _TOY.replace("GROUP = 32;", f"GROUP = {group};"))
    operands = _toy_matrix(8, 64, seed=2, group=group)
    exact, _ = testing.tile_rows(face, operands, 64, range(8), mx.float32)
    assert _same(exact, _toy_decode(*operands, group=group))


# A face that keeps state between calls (rule 8): params, or fetch, answers
# the first group or run it was asked for; fetch, the run after the one it
# gave last.
_KEEPS_STATE = {
    "params": [
        (
            "    const device float* r;\n",
            "    const device float* r;\n    mutable int first = -1;\n",
        ),
        (
            "      return {g[group] * r[0]};",
            "      first = first < 0 ? group : first;\n      return {g[first] * r[0]};",
        ),
    ],
    "fetch": [
        (
            "    const device float* r;\n",
            "    const device float* r;\n    mutable int first = -1;\n",
        ),
        (
            "      return c[run];",
            "      first = first < 0 ? run : first;\n      return c[first];",
        ),
    ],
    "fetch_advancing": [
        ("    const device uint2* c;\n", "    mutable const device uint2* c;\n"),
        ("      return c[run];", "      return *c++;"),
    ],
}


@metal
@pytest.mark.parametrize("member", _KEEPS_STATE)
def test_tile_rows_show_a_member_that_keeps_state(member):
    source = _TOY
    for old, new in _KEEPS_STATE[member]:
        assert source.count(old) == 1
        source = source.replace(old, new)
    operands = _toy_matrix(8, 64, seed=4)
    exact, _ = testing.tile_rows(
        _toy(f"state_{member}", source), operands, 64, range(8), mx.float32
    )
    assert not _same(exact, _toy_decode(*operands))


def _sentences(source) -> list[str]:
    """The sentences of the static_asserts in ``source``, each from its
    adjacent string literals."""
    asserts = re.findall(r"static_assert\(\s*[^;]*?,\s*((?:\"[^\"]*\"\s*)+)\);", source)
    return ["".join(re.findall(r"\"([^\"]*)\"", literals)) for literals in asserts]


def test_the_page_lists_every_sentence_of_the_tile_skeleton():
    section = _PAGE.split("\n## The tile face\n")[1].split("\n## ")[0]
    listed = re.findall(r"^- `(tile face: [^`]*)`$", section, re.M)
    assert listed == _sentences(kernels._TILE_FACE)
    assert len(listed) == 11 and all(s.startswith("tile face: ") for s in listed)


# --- testing.tile_rows ------------------------------------------------------


def test_tile_rows_refuses_what_it_cannot_read():
    operands = _toy_operands()
    face = _toy("refusals")
    for k in (0, 12, -8, 8.0, True, 1 << 31):
        with pytest.raises(ValueError, match=f"k={k} is not one run or more of 8"):
            testing.tile_rows(face, operands, k, range(8), mx.float32)
    for dtype in (mx.int32, mx.bool_, mx.float64, "bfloat16"):
        with pytest.raises(ValueError, match="is not a float type of the GPU"):
            testing.tile_rows(face, operands, 64, range(8), dtype)
    for rows in ([], [-1], [0.0], ["0"], [True], [0, False]):
        with pytest.raises(ValueError, match="are not one row index or more"):
            testing.tile_rows(face, operands, 64, rows, mx.float32)
    with pytest.raises(ValueError, match="weight, group_scales, row_scales; operands"):
        testing.tile_rows(face, operands[:2], 64, range(8), mx.float32)
    half = replace(face, dtypes=(mx.float16,))
    with pytest.raises(ValueError, match="exact for mlx.core.float16, not mlx.core"):
        testing.tile_rows(half, operands, 64, range(8), mx.bfloat16)
    for rows in (4, 7):
        codes, group_scales, row_scales = _toy_matrix(rows, 64)
        with pytest.raises(
            ValueError, match=f"toy_refusals: row_scales has {rows} elements"
        ):
            testing.tile_rows(
                face, (codes, group_scales, row_scales), 64, range(rows), mx.float32
            )
    with pytest.raises(ValueError, match="is not a Tile"):
        testing.tile_rows(
            affine.face(4, 64, True, mx.bfloat16), operands, 64, [0], mx.float32
        )


# K of whole groups, of a last group that is partial (40: 32 + 8; 8: one run
# of a group of 32), and of many groups.
@metal
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
@pytest.mark.parametrize("k", [64, 40, 8, 1024])
def test_tile_rows_are_the_decode_in_float32_and_rounded_once(k, dtype):
    operands = _toy_matrix(16, k, seed=k)
    want = _toy_decode(*operands)
    exact, rounded = testing.tile_rows(_toy("rows"), operands, k, range(16), dtype)
    assert exact.dtype == mx.float32 and rounded.dtype == dtype
    assert exact.shape == rounded.shape == (16, k)
    assert _same(exact, want)
    assert _same(rounded, want.astype(dtype))


@metal
def test_tile_rows_read_the_rows_they_are_given():
    # A stack of 3 matrices of 8 rows: row n of matrix e is e * 8 + n.
    codes, group_scales, row_scales = _toy_matrix(24, 64, seed=7)
    stack = (
        codes.reshape(3, 8, 64),
        group_scales.reshape(3, 8, 2),
        row_scales.reshape(3, 8),
    )
    want = _toy_decode(codes, group_scales, row_scales)
    rows = [23, 0, 17, 17, 8, 5]  # the last, the first, out of order, twice
    exact, _ = testing.tile_rows(_toy("stack"), stack, 64, rows, mx.float32)
    assert _same(exact, want[mx.array(rows)])
    # Fewer than 8 rows: MLX binds their indices as constants.
    exact, _ = testing.tile_rows(_toy("stack"), stack, 64, [23], mx.float32)
    assert _same(exact, want[23:])


@metal
def test_two_threads_read_one_tile_face_at_once():
    operands = _toy_matrix(16, 256, seed=3)
    want = _toy_decode(*operands)
    # Arrays a thread reads are evaluated: a lazy one is the stream's of the
    # thread that made it.
    mx.eval(*operands, want)
    face = _toy("threads")  # compiled by whichever thread comes first

    def read(_):
        exact, rounded = testing.tile_rows(face, operands, 256, range(16), mx.bfloat16)
        mx.eval(exact, rounded)
        return exact, rounded

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(read, range(4)))
    for exact, rounded in results:
        assert _same(exact, want) and _same(rounded, want.astype(mx.bfloat16))


# --- The page's example -----------------------------------------------------


def _fenced(page) -> list[tuple[str, str]]:
    lines = iter(page.splitlines())
    return [
        (
            line.strip()[3:],
            "\n".join(takewhile(lambda inner: inner.strip() != "```", lines)),
        )
        for line in lines
        if line.strip().startswith("```")
    ]


_BLOCKS = _fenced(_PAGE)


def test_the_pages_tile_example_is_the_toy():
    (source,) = (
        text
        for language, text in _BLOCKS
        if language == "cpp" and text.startswith("namespace omlx_myformat_b8g32 {")
    )
    assert source + "\n" == _TOY.replace("omlx_toy_KEY", "omlx_myformat_b8g32")
    (face,) = (
        text
        for language, text in _BLOCKS
        if language == "python" and text.startswith("Tile(")
    )
    assert eval(face, {"Tile": formats.Tile, "METAL": _TOY}) == formats.Tile(
        key="myformat_b8g32",
        source=_TOY,
        type="omlx_myformat_b8g32::Tile",
        tensors=("group_scales", "row_scales"),
    )


@metal
def test_the_pages_decode_is_the_tile_faces_weights():
    (decode,) = (
        text
        for language, text in _BLOCKS
        if language == "python" and "def decode(" in text and "group_scales" in text
    )
    scope = {"mx": mx}
    exec("class Format:\n" + decode, scope)
    codes, group_scales, row_scales = _toy_matrix(8, 40, seed=5)
    tensors = {"weight": codes, "group_scales": group_scales, "row_scales": row_scales}
    want = scope["Format"]().decode(tensors, 8, 32)
    exact, _ = testing.tile_rows(
        _toy("page"), list(tensors.values()), 40, range(8), mx.float32
    )
    assert _same(exact, want) and _same(
        want, _toy_decode(codes, group_scales, row_scales)
    )


# --- Affine -----------------------------------------------------------------


def _layer(k, n, bits, group_size, dtype=mx.bfloat16, experts=None, seed=0):
    """An affine layer of MLX's: a matrix, or a stack of ``experts``."""
    mx.random.seed(seed)
    shape = (n, k) if experts is None else (experts, n, k)
    weight = (mx.random.normal(shape) * 0.05).astype(dtype)
    layer = nn.QuantizedLinear(k, n, bias=False, group_size=group_size, bits=bits)
    layer.weight, layer.scales, layer.biases = mx.quantize(
        weight, group_size=group_size, bits=bits
    )
    return layer


def _float32_decode(layer):
    """MLX's dequantize computed in float32: over float32 copies of the scales
    and the biases (with ``dtype=mx.float32`` MLX 0.32.4 computes in the
    scales' type and widens the result)."""
    return mx.dequantize(
        layer.weight,
        layer.scales.astype(mx.float32),
        layer.biases.astype(mx.float32),
        group_size=layer.group_size,
        bits=layer.bits,
    )


_LAYOUTS = [(bits, gs) for bits in (4, 8) for gs in (32, 64, 128)]
_DTYPES = [mx.bfloat16, mx.float16]


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("bits, group_size", _LAYOUTS)
@pytest.mark.parametrize("experts", [None, 3])
def test_affine_gives_its_4_and_8_bit_layers_a_tile_face(
    bits, group_size, dtype, experts
):
    layer = _layer(256, 16, bits, group_size, dtype, experts)
    face = formats.tile(layer, 256, 16)
    suffix = {mx.bfloat16: "bf16", mx.float16: "fp16"}[dtype]
    metal_type = {mx.bfloat16: "bfloat16_t", mx.float16: "float16_t"}[dtype]
    assert face is affine.tile_face(bits, group_size, dtype)
    assert face == formats.Tile(
        key=f"affine_tile_b{bits}g{group_size}_{suffix}",
        source=affine._tile_source(bits, group_size),
        type=f"omlx_affine_tile_b{bits}g{group_size}::Tile<{metal_type}>",
        tensors=("scales", "biases"),
        dtypes=(dtype,),
    )
    # One source for both types of scales.
    other = affine.tile_face(bits, group_size, _DTYPES[dtype == mx.bfloat16])
    assert other.source == face.source and other.key != face.key


def _with(layer, **arrays):
    for name, array in arrays.items():
        layer[name] = array
    return layer


# What affine has no tile face for, each as a layer of K 256 and N 16 that
# differs in that alone from one it has a face for.
_NO_TILE = [
    "2 bits",
    "3 bits",
    "5 bits",
    "6 bits",
    "groups of 16",
    "float32 scales",
    "biases of another dtype",
    "no scales",
    "no biases",
    "codes of another type",
    "a stack of stacks",
    "scales of other rows",
    "biases of other rows",
    "no weight",
    "biases that are not an array",
]


def _no_tile(name):
    bits = {"2 bits": 2, "3 bits": 3, "5 bits": 5, "6 bits": 6}.get(name, 4)
    dtype = mx.float32 if name == "float32 scales" else mx.bfloat16
    layer = _layer(256, 16, bits, 64, dtype)
    if name == "groups of 16":
        layer.group_size = 16
        return _with(
            layer,
            scales=mx.repeat(layer.scales, 4, axis=-1),
            biases=mx.repeat(layer.biases, 4, axis=-1),
        )
    if name == "biases of another dtype":
        return _with(layer, biases=layer.biases.astype(mx.float16))
    if name in ("no scales", "no biases", "no weight"):
        del layer[name[3:]]
    if name == "biases that are not an array":
        return _with(layer, biases=None)
    if name == "codes of another type":
        return _with(layer, weight=layer.weight.view(mx.int32))
    if name == "a stack of stacks":
        return _with(
            layer, **{n: layer[n][None, None] for n in ("weight", "scales", "biases")}
        )
    if name == "scales of other rows":
        return _with(layer, scales=layer.scales[:8])
    if name == "biases of other rows":
        return _with(layer, biases=layer.biases[:8])
    return layer


# No face, and no defect of the built-in format either: the test fails at its
# teardown on one (tests/conftest.py, dot_defects).
@pytest.mark.parametrize("name", _NO_TILE)
def test_affine_has_no_tile_face_for_what_it_does_not_read(name):
    assert formats.tile(_no_tile(name), 256, 16) is None


def test_affine_has_no_tile_face_for_a_k_that_ends_inside_a_group():
    layer = _layer(256, 16, 4, 64)
    # 224 inputs: 28 words of codes, 3 whole groups and half of a fourth.
    for name, array in (("weight", 28), ("scales", 3), ("biases", 3)):
        layer[name] = layer[name][:, :array]
    assert formats.tile(layer, 224, 16) is None
    layer = _layer(192, 16, 4, 64)
    assert formats.tile(layer, 192, 16) is not None


def test_affine_has_no_tile_face_for_fewer_than_8_groups():
    # MLX binds an input of fewer than 8 elements as constants.
    assert formats.tile(_layer(64, 7, 4, 64), 64, 7) is None
    assert formats.tile(_layer(64, 8, 4, 64), 64, 8) is not None


def test_affine_has_no_tile_face_for_a_k_or_n_that_is_not_the_layers():
    layer = _layer(256, 16, 4, 64)
    assert formats.tile(layer, 256, 16) is not None
    for k, n in ((128, 16), (512, 16), (256, 8), (256, 32), (224, 16)):
        assert formats.tile(layer, k, n) is None


def test_the_affine_tile_face_is_exact_for_its_scales_dtype_only():
    layer = _layer(256, 16, 4, 64)
    face = formats.tile(layer, 256, 16)
    assert face.exact_for(mx.bfloat16)
    for dtype in (mx.float16, mx.float32):
        assert not face.exact_for(dtype)
        with pytest.raises(ValueError, match="exact for mlx.core.bfloat16"):
            testing.tile_rows(
                face, [layer.weight, layer.scales, layer.biases], 256, [0], dtype
            )


@pytest.mark.parametrize(
    "bits, group_size", [(2, 64), (3, 64), (5, 64), (4, 16), (8, 256)]
)
def test_a_tile_face_built_by_hand_is_one_affine_has(bits, group_size):
    with pytest.raises(
        ValueError, match=f"no runs of {bits} bits in groups of {group_size}"
    ):
        affine.tile_face(bits, group_size, mx.bfloat16)


def test_no_affine_tile_face_for_float32_scales():
    with pytest.raises(ValueError, match="no scales of mlx.core.float32"):
        affine.tile_face(4, 64, mx.float32)


# Each width's source holds its own run's type and code alone (what is not
# used is not hashed at every launch).
@pytest.mark.parametrize(
    "bits, kept, left",
    [(4, "uint code_at(uint q", ("uint2", "0xFFu")), (8, "packed_uint2", ("0xFu;",))],
)
def test_each_affine_tile_width_has_its_runs_alone(bits, kept, left):
    source = affine.tile_face(bits, 64, mx.bfloat16).source
    assert kept in source and source.count("code_at(") == 2
    assert not any(text in source for text in left)


@metal
def test_the_affine_tile_face_does_not_compile_for_a_k_that_is_not_whole_groups():
    layer = _layer(256, 16, 4, 64)
    face = formats.tile(layer, 256, 16)
    operands = [layer[name] for name in ("weight", *face.tensors)]
    with pytest.raises(RuntimeError) as failure:
        mx.eval(testing.tile_rows(face, operands, 96, [0], mx.bfloat16))
    message = str(failure.value)
    assert (
        "affine tile face: K must be whole groups; AffineFormat.tile gives a K its face"
    ) in message
    assert message.count("error:") == 1


def _affine_rows(layer, k, dtype):
    face = formats.tile(layer, k, layer.weight.shape[-2])
    rows = layer.weight.size * 32 // layer.bits // k
    operands = [layer[name] for name in ("weight", *face.tensors)]
    return testing.tile_rows(face, operands, k, range(rows), dtype)


@metal
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("bits, group_size", _LAYOUTS)
def test_affine_tile_weights_are_mlx_dequantize_in_float32(bits, group_size, dtype):
    # Every row of a stack of 3 matrices: the last one's last rows included.
    layer = _layer(512, 24, bits, group_size, dtype, experts=3, seed=bits + group_size)
    want = _float32_decode(layer).reshape(72, 512)
    exact, rounded = _affine_rows(layer, 512, dtype)
    assert _same(exact, want)
    assert _same(rounded, want.astype(dtype))


def _finite(shape, dtype, seed):
    """Random bits of ``dtype``, their exponents kept below the largest: no
    infinity, no NaN."""
    mx.random.seed(seed)
    bits = mx.random.randint(0, 1 << 16, shape).astype(mx.uint16)
    top = {mx.bfloat16: 0x7F80, mx.float16: 0x7C00}[dtype]  # the exponent's bits
    bits = mx.where((bits & top) == top, bits - (top & -top), bits)
    return bits.view(dtype)


def _canary(name, bits, dtype, n=16, k=256):
    """An affine layer of ``n`` rows of K inputs whose arrays a test of rules
    1 and 2 reads: every byte at its largest, random bytes, zeros, and fp16's
    finite products past its largest value."""
    layer = _layer(k, n, bits, 64, dtype)
    words, groups = (n, k * bits // 32), (n, k // 64)
    largest = {mx.bfloat16: 0x7F7F, mx.float16: 0x7BFF}[dtype]
    if name == "largest":
        weight = mx.full(words, 0xFFFFFFFF, mx.uint32)
        scales = mx.full(groups, largest, mx.uint16).view(dtype)
        biases = mx.full(groups, largest | 0x8000, mx.uint16).view(dtype)
    elif name == "random":
        mx.random.seed(11)
        weight = mx.random.randint(0, 1 << 16, (*words, 2)).astype(mx.uint32)
        weight = weight[..., 0] | (weight[..., 1] << 16)
        scales, biases = _finite(groups, dtype, 12), _finite(groups, dtype, 13)
    elif name == "zeros":
        weight = mx.zeros(words, mx.uint32)
        scales, biases = mx.zeros(groups, dtype), mx.zeros(groups, dtype)
    else:  # 32768 * 3 - 65504 = 32800: in fp16, finite computed in float32 only
        weight = mx.full(words, 0x33333333 if bits == 4 else 0x03030303, mx.uint32)
        scales = mx.full(groups, 32768.0, dtype)
        biases = mx.full(groups, -65504.0, dtype)
    return _with(layer, weight=weight, scales=scales, biases=biases)


@metal
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("name", ["largest", "random", "zeros", "past_fp16"])
def test_affine_tile_weights_hold_on_canaries(name, bits, dtype):
    layer = _canary(name, bits, dtype)
    want = _float32_decode(layer)
    exact, rounded = _affine_rows(layer, 256, dtype)
    assert _same(exact, want) and _same(rounded, want.astype(dtype))
    if name == "zeros":
        assert not mx.any(exact).item()
    if name == "past_fp16" and dtype == mx.float16:
        assert mx.all(exact == 32800.0).item() and mx.all(rounded == 32800.0).item()
