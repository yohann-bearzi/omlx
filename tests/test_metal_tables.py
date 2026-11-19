# SPDX-License-Identifier: Apache-2.0
"""No Metal of oMLX's declares a table by hand: a face that declares an array
with static storage is refused when it is made, and oQ writes no checkpoint
for it. Every kernel built over faces is checked here, and every Metal text
in the repository for arrays declared `static` or `constant`, with the
exceptions below and their reasons (docs/quant-formats.md, Tables)."""

from __future__ import annotations

import ast
import re
import sys
from collections import Counter
from dataclasses import dataclass, field, fields
from pathlib import Path

import mlx.core as mx
import pytest

from omlx import formats
from omlx.formats import Dot, Table, Tile, affine, kernels, testing
from omlx.formats import linear as format_linear
from omlx.patches import m5_gather_qmm_nax as nax
from omlx.patches import moe_verify_gather as gather
from omlx.patches import qwen35_moe_routed_decode as routed
from tests.test_formats_tables import (
    LUT,
    _capture_nax_without_mlx_headers,
    _toy,
    tabled,
    tabled_tile,
)

ROOT = Path(__file__).resolve().parents[1]


def hand_tables(text):
    return kernels.hand_tables(text)


# Tables written by hand that stay, each with its reason. One that matches
# nothing any more fails the test: no exception outlives its table.
ALLOWED = {
    ("omlx/patches/m5_gather_qmm_a8.py", "kE2M1x2"): (
        "upstream 624cff0, unchanged: the M5 Max A8 gather's FP4 pair table "
        "(UPSTREAM-BUGS E8)"
    ),
    ("omlx/patches/deepseek_v41/decode_moe.py", "FP4_VALUES"): (
        "upstream 624cff0, unchanged, and never read (UPSTREAM-BUGS E8)"
    ),
    ("omlx/custom_kernels/lowbit/csrc/t5_quantized.h", "T5_TO_B4"): (
        "upstream 624cff0, unchanged: lowbit_qmv.metal stages it, which "
        "t5_quantized.h then does not read (UPSTREAM-BUGS E8)"
    ),
}

_MARK = re.compile(r"//\s*omlx: staged (\w+)")


def _strings(tree):
    """The string constants of a module, its docstrings left out, with the
    literal parts of each f-string joined around its fields (read as 0)."""
    skip = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                skip.add(id(first.value))
        if isinstance(node, ast.JoinedStr):
            skip.update(id(value) for value in node.values)
            yield "".join(
                v.value if isinstance(v, ast.Constant) else "0" for v in node.values
            )
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in skip
        ):
            yield node.value


def _staged_natively(text, line, name):
    """A native file's table that it marks as staged: the line above says
    ``// omlx: staged <name>``, and the file copies it into threadgroup
    memory."""
    lines = text.split("\n")
    above = _MARK.search(lines[line - 2]) if line >= 2 else None
    if not above or above[1] != name:
        return False
    code = kernels._metal_code(text)
    for body in re.finditer(r"\bkernel\b[^;{}]*\{", code):
        start = body.end()
        depth = 1
        for brace in re.finditer(r"[{}]", code[start:]):
            depth += 1 if brace[0] == "{" else -1
            if not depth:
                break
        if depth:
            continue
        kernel = code[start : start + brace.start()]
        for array in re.finditer(r"\bthreadgroup\s+[\w:]+\s+(\w+)\s*\[[^;]+;", kernel):
            copy = re.search(
                rf"\b{array[1]}\s*\[[^;]+?\]\s*=\s*{name}\s*\[[^;]+?\]\s*;",
                kernel[array.end() :],
            )
            if copy and re.search(
                r"\bthreadgroup_barrier\s*\(\s*mem_flags::mem_threadgroup\s*\)",
                kernel[array.end() + copy.end() :],
            ):
                return True
    return False


def _found():
    """Keep every occurrence, including duplicate names in a single file."""
    found = []
    for path in sorted((ROOT / "omlx").rglob("*.py")):
        for text in _strings(ast.parse(path.read_text())):
            if ";" in text:
                for _, name, shown in hand_tables(text):
                    found.append((str(path.relative_to(ROOT)), name, shown))
    for path in sorted((ROOT / "omlx" / "custom_kernels").rglob("*")):
        if path.suffix not in (".metal", ".h", ".hpp", ".cpp", ".mm"):
            continue
        text = path.read_text()
        for line, name, shown in hand_tables(text):
            if not _staged_natively(text, line, name):
                found.append((str(path.relative_to(ROOT)), name, shown))
    for path in sorted((ROOT / "docs").rglob("*.md")):
        for block in re.findall(r"```(?:cpp|metal)\n(.*?)```", path.read_text(), re.S):
            for _, name, shown in hand_tables(block):
                found.append((str(path.relative_to(ROOT)), name, shown))
    return found


def test_no_metal_of_omlx_declares_a_table_by_hand():
    found = _found()
    assert Counter((path, name) for path, name, _ in found) == Counter(
        ALLOWED.keys()
    ), found


def test_only_the_staging_writes_its_section():
    marker = kernels._STAGED
    holders = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "omlx").rglob("*.py")
        if marker in path.read_text()
    ]
    assert holders == ["omlx/formats/kernels.py"]


def test_repository_scan_finds_each_declaration_even_when_names_repeat(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    package = tmp_path / "omlx"
    package.mkdir()
    (package / "example.py").write_text(
        'FIRST = "constant float LUT[2] = {1, 2};"\n'
        'SECOND = "static float LUT[4];"\n'
    )
    assert len(_found()) == 2


def test_a_native_table_passes_when_it_is_marked_and_staged():
    text = (
        "// omlx: staged LUT\n"
        "constant constexpr uint LUT[4] = {1, 2, 3, 4};\n"
        "kernel void k(device uint* y) {\n"
        "  threadgroup uint lut[4];\n"
        "  for (uint i = 0; i < 4; ++i) { lut[i] = LUT[i]; }\n"
        "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "}\n"
    )
    ((line, name, _),) = hand_tables(text)
    assert _staged_natively(text, line, name)
    assert not _staged_natively(text.replace("staged LUT", "staged X"), line, name)
    assert not _staged_natively(text.replace("= LUT[i]", "= 1"), line, name)
    assert not _staged_natively(text.replace("= LUT[i]", "= OTHER[i]"), line, name)
    assert not _staged_natively(text.replace("lut[i] =", "y[i] ="), line, name)
    assert not _staged_natively(text.replace("threadgroup_barrier", "f"), line, name)
    assert not _staged_natively(
        text.replace("= LUT[i];", "= 0; // = LUT[i];"), line, name
    )
    assert not _staged_natively(
        text.replace("mem_threadgroup", "mem_device"), line, name
    )
    other = text.replace("  for", "}\nkernel void other(device uint* lut) {\n  for")
    assert not _staged_natively(other, line, name)
    early = text.replace(
        "  for", "  threadgroup_barrier(mem_flags::mem_threadgroup);\n  for"
    )
    early = early.replace("  threadgroup_barrier(mem_flags::mem_threadgroup);\n}", "}")
    assert not _staged_natively(early, line, name)
    assert not _staged_natively(text.replace("// omlx: staged LUT\n", ""), 1, name)


@pytest.mark.parametrize(
    "source,names",
    [
        ("constant constexpr float LEVELS[] = {1.0f};", ["LEVELS"]),
        ("static constant constexpr float v[4][16] = {{1}};", ["v"]),
        ("struct S { static constant constexpr float v[2] = {1, 2}; };", ["v"]),
        ("alignas(256) constant constexpr float L[4] = {1, 2};", ["L"]),
        ("constant float *p = 0, LUT[16] = {1};", ["LUT"]),
        ("static float D[4];", ["D"]),
        ("void f() { static float LUT[4]; }", ["LUT"]),
        ("typedef constant float lut_t[4];", []),
        ("constant float4 v = float4(1, 2, 3, 4);", []),
        ("constant metal::array<float, 16> L = {1, 2};", []),
        ("constant constexpr S L{};", []),
        ("using table_t = float[16]; constant table_t L = {1, 2};", []),
        ("constant constexpr auto L = {1, 2};", []),
        ("const metal::conditional_t<B, const device T*, device U*> p = x;", []),
        ('const constant char* m = "x; constant float L[4] = {1};";', []),
        ("constant float (*p)[4] = values;", []),
        ("void f() { static int y = LUT[i]; }", []),
        ("for (int i = 0; i<N; ++i) {}\nconstant float L[4] = {1};", ["L"]),
        ("template <typename T> void f(const device T* x, device T* y) {}", []),
        ("kernel void k() { threadgroup float part[16]; }", []),
        ("#define X constant float LUT[4];", []),
        ("constant float LUT alignas(16)[4] = {1};", ["LUT"]),
    ],
)
def test_what_counts_as_a_table_written_by_hand(source, names):
    assert [name for _, name, _ in hand_tables(source)] == names


def test_the_repository_scan_reads_static_and_constant_arrays():
    # A repository scan keeps the qualifier rule (lead, 11-19 22:55): the
    # module's docstring says which arrays it reads.
    assert [name for _, name, _ in hand_tables("static float LUT[4];")] == ["LUT"]
    assert [name for _, name, _ in hand_tables("constant int LUT[4];")] == ["LUT"]
    assert hand_tables("constexpr float LUT[4] = {1};") == ()
    assert "`static` or `constant`" in " ".join(__doc__.split())


@pytest.mark.parametrize(
    "source",
    [
        "constant unsigned int M = 1;",
        "constant signed char M = 1;",
        "constant long long M = 1;",
        "constant constexpr auto M = int{4};",
        "constant constexpr auto M = 1'024;",
        "using Index = uint; constant Index M = 1;",
        "struct Face { int f() const constant { return 4; } };",
        "struct Face { constant static int M = K < 512 ? 8 : 16; struct L {}; };",
        "metal::conditional_t<(B > 4), const device uint*, const device uchar*> p;",
        "constant float4 v = {1, 2, 3, 4};",
        "constant int s = 1, &r = s;",
        "constant float (&r)[4] = values;",
        "constant float (*p)[4] = values;",
        "typedef unsigned int Index; constant Index i{4};",
        "constant metal::vec<float, 4> v{1, 2, 3, 4};",
        "template <class T> struct S { static constant T value = v; };",
        "constant float f(int x);",
        "constant constexpr auto N{4};",
        "using V = metal::vec<float, 4>; constant V v{1,2,3,4};",
        "auto v = static_cast<const device float*>(p);",
        "using X = metal::conditional_t<(p->n > 4), device uint*, uint*>;",
        "struct F { int a, b; F(); }; F::F() : a{0}, b{1} {}",
        "void f() { static int y = LUT[i]; }",
    ],
)
def test_a_valid_face_accepts_scalar_vector_and_function_declarations(source):
    Dot(key="legal", source=source, type="F", tensors=())


@pytest.mark.parametrize(
    "source",
    [
        "constant constexpr float LUT[2 * N] = {1.0f};",
        "constant float s = 1, LUT[4 * 4] = {1};",
        "constant const float* LUT[4] = {p};",
        "template <int B> static constant float LUT[B] = {1};",
        "constant int A = 2 > 1 ? 1 : 0, LUT[16] = {1, 2};",
        "alignas(0x1'0) constant float LUT[0x1'0] = {};",
        "int n = 1'0; constant float LUT[4] = {1};",
        "constant float v = 0, *p = nullptr, LUT[4] = {1};",
        'const char* message = "a "quote""; constant float LUT[4] = {};',
        "static int x = a < b, LUT[4], y = c > d;",
        "template <bool B = 1 < 2> constant float LUT[4] = {1};",
    ],
)
def test_a_table_is_reported_by_its_declarator_name(source):
    assert [name for _, name, _ in hand_tables(source)] == ["LUT"]
    with pytest.raises(ValueError, match=r"declares LUT.*Tile\("):
        Tile(key="bad", source=source, type="F", tensors=())


def test_a_dot_reports_tables_in_its_shared_source():
    message = r"line 2 of its shared source declares LEVELS"
    with pytest.raises(ValueError, match=message):
        Dot(
            key="shared",
            source="struct Face {};",
            shared="// shared rows\nconstant float LEVELS[2] = {1, 2};",
            type="Face",
            tensors=(),
            vpt=8,
        )


def test_validation_follows_the_source_parts_a_face_contributes():
    source_field = next(part for part in fields(Dot) if part.name == "source")

    @dataclass(frozen=True)
    class Extended(Dot):
        extra: str = field(default="", metadata=source_field.metadata)

    face = Extended("extra", "struct F {};", "F", (), extra="using Aux = int;")
    builder = kernels.Matrices()
    builder.add("m", face)
    assert face.extra in builder.header
    with pytest.raises(ValueError, match=r"line 2 of its extra source declares LUT"):
        Extended("extra", face.source, "F", (), extra="// extra\ndevice int LUT[8];")


def test_a_face_s_empty_main_source_is_still_cited_in_the_header():
    builder = kernels.Matrices()
    builder.add("m", Dot("empty", "", "F", ()))
    assert '\n#line 1 "<quant format empty>"\n' in builder.header


@pytest.mark.parametrize(
    "source",
    [
        "extern const float LUT[4];",
        "namespace N { constexpr float LUT[4] = {1}; }",
        "void f() { static constexpr float LUT[4] = {1}; }",
        "void f() { constexpr float LUT[4] = {1}; }",
        "struct F { static constexpr float LUT[4] = {1}; };",
        "void f(bool b) { if (b) {} else {} static constant int LUT[4]; }",
        "void f() { try {} catch (...) {} static constexpr int LUT[4] = {1}; }",
        "struct F { F& operator=(const F&) { return *this; }"
        " static constant int LUT[4]; };",
        "void f() { static const device float *p = nullptr, *LUT[4] = {p}; }",
        "void f() { static int x = 1, LUT[4] = {x}; }",
        "constant float LUT[4];",
        "float LUT[4];",
        "namespace n { float LUT[4]; }",
        'extern "C" { float LUT[4]; }',
        'extern "C" { void f(); }\nextern "C" { float LUT[4]; }',
    ],
)
def test_a_face_refuses_arrays_with_static_storage(source):
    with pytest.raises(ValueError, match=r"its source declares LUT "):
        Dot("storage", source, "F", ())


@pytest.mark.parametrize(
    "source",
    [
        "class F { public: static constant float LUT[4] = {1}; };",
        "struct F { private: static constexpr float LUT[4] = {1}; };",
        "void f(int x) { switch (x) { case 0: static constant float LUT[4] = {1}; } }",
        "void f() { done: static constant float LUT[4] = {1}; }",
        "struct F { F& operator+=(int x) { return *this; }"
        " static constant float LUT[4] = {1}; };",
        "struct F { bool operator!=(const F& o) const { return false; }"
        " static constant float LUT[4] = {1}; };",
        "struct F { int v; }; inline F& operator+=(thread F& a, int b)"
        " { a.v += b; return a; } constant float LUT[4] = {1};",
        "struct F { int a, b; F() : a{0}, b{1} {}"
        " static constant float LUT[4] = {1}; };",
        "struct F { int a, b; F(); }; F::F() : a{0}, b{1} {}"
        " constant float LUT[4] = {1};",
    ],
)
def test_a_face_refuses_tables_after_specifiers_labels_and_operators(source):
    with pytest.raises(ValueError, match=r"its source declares LUT "):
        Dot("access", source, "F", ())


@pytest.mark.parametrize(
    "source",
    [
        "enum Mode { A = 1, B = 2 }; constant constexpr Mode M{A};",
        "enum class Mode : int { A = 1 };",
        "enum { A = 1 };",
        "constant enum Mode { A = 1 } M{A};",
        "enum struct Mode { A = 1 }; constant Mode M{Mode::A};",
        "enum class Mode : int { A = 1 }; constant Mode M{Mode::A};",
        "constant enum struct Mode : int { A = 1 } M{Mode::A};",
    ],
)
def test_a_face_accepts_namespace_enums(source):
    Dot("enum_ok", source, "F", ())


@pytest.mark.parametrize(
    "source",
    [
        "constant enum Mode { A, B } LUT[4] = {A};",
        "enum Mode { A = 1 }; constant Mode LUT[4] = {A};",
    ],
)
def test_a_face_refuses_enum_arrays_with_static_storage(source):
    with pytest.raises(ValueError, match=r"its source declares LUT "):
        Dot("enum_table", source, "F", ())


@pytest.mark.parametrize(
    "source",
    [
        "enum Mode { A = 1 }; using M = Mode; constant M x{A};",
        "enum Mode { A = 1 }; typedef Mode M; constant M x{A};",
        "typedef enum { A = 1 } Mode; constant Mode x{A};",
        "using P = const device float*; constant P p{nullptr};",
    ],
)
def test_a_face_accepts_aliased_enum_and_pointer_scalars(source):
    Dot("alias", source, "F", ())


@pytest.mark.parametrize(
    "source, names",
    [
        ("struct F { float a[4]; }; void f() { float a[4]; }", []),
        ("union U { float a[4]; }; void f() { U value{}; }", []),
        ("constant bool HAS [[function_constant(0)]];", []),
        ("void f() { const device float *p = nullptr, *rows[4] = {p}; }", []),
        ("constant float (*p)[4], LUT[4];", ["LUT"]),
        (
            "template <template <class T = int> class C> "
            "struct F { static constant int LUT[4]; };",
            ["LUT"],
        ),
        (
            "struct thread_position_in_grid {}; "
            "static_assert(sizeof(thread_position_in_grid));"
            " const device float *LUT[4];",
            ["LUT"],
        ),
    ],
)
def test_scope_and_declarator_guards(source, names):
    assert [name for _, name, _ in kernels.hand_tables(source, face=True)] == names


# Neon's findings on 0689b59 (reviews/pre-rtables-0689b59.md) under the
# lead's ruling of 11-20: the scan is a speed lint, not a parser. It refuses
# the tables authors write, never refuses code that is not a table, and the
# spellings it does not read are listed below and in the guide.


@pytest.mark.parametrize(
    "source",
    [
        # 1: an '=' at another brace level before the table
        "struct F { bool operator!=(const F& o) const"
        " { static constexpr float LUT[4] = {1}; return false; } };",
        "struct F { F& operator=(const F& o)"
        " { static constant float LUT[4] = {1}; return *this; } };",
        "struct F { F& operator+=(int x)"
        " { static constexpr float LUT[4] = {1}; return *this; } };",
        "constant enum Mode { A = 1 } LUT[4] = {A};",
        "void f() { static enum Mode { A = 1 } LUT[4] = {A}; }",
        "constant float s = metal::sqrt(2.0f), LUT[4] = {1};",
        # 3: a storage qualifier before an inline struct or union
        "void f() { static struct S { float v; } LUT[4] = {{1}}; }",
        "struct F { static constant struct { float a, b; } LUT[4] = {{1, 2}}; };",
        "void f() { static union U { float a; uint b; } LUT[4] = {{1}}; }",
        "void f() { static struct alignas(16) S { float v; } LUT[4] = {}; }",
        "constant decltype(1.0f) LUT[4] = {1};",
        "void f() { static decltype(foo()) a{1}, LUT[4] = {1}; }",
        # 4: a namespace after a function body is namespace scope
        "inline float g() {}\nnamespace N { float LUT[4]; }",
        "namespace A { void g() {} }\nnamespace B { float LUT[4]; }",
        # pre-rtables-4a8f23c: GNU attributes, a storage class after a condition
        "void f() { static struct __attribute__((packed)) S { float v; }"
        " LUT[4] = {{1}}; }",
        "constant float __attribute__((aligned(16))) LUT[4] = {1};",
        "void f() { if constexpr (true) static float LUT[4] = {1}; }",
        # pre-rtables-9ff8169: '!=' inside a template argument
        "template <bool B> struct Box { float v; };"
        " constant Box<K != 0> LUT[4] = {};",
        "static struct F : Base<K != 0> { float v; } LUT[4] = {};",
        "inline namespace N { float LUT[4]; }",
    ],
)
def test_a_face_refuses_the_tables_authors_write(source):
    with pytest.raises(ValueError, match=r"its source declares LUT "):
        Dot("written", source, "F", ())


@pytest.mark.parametrize(
    "source",
    [
        # 4: an empty block ends its statement
        "void f() { if constexpr (true) {} float a[4]; }",
        "struct F { constexpr F() {} float a[4]; };",
        "struct F { static void reset() {} float a[4]; };",
        "namespace N { constant int K = 4; }\nstruct F { float a[4]; };",
        # 6: array parameters and subscript operators
        "inline float sums(const thread float x[4]) { return x[0]; }",
        "struct F { static float sums(const thread float x[4]) { return x[0]; } };",
        "struct F { float operator[](int i) const; };\n"
        "float F::operator[](int i) const { return 0.0f; }",
        "struct F { constexpr float operator[](int i) const { return 0.0f; } };",
        "struct F { constexpr float operator[](int i) const; };",
        # an indexed store after a condition, an alias of an array type
        "void f(thread float* x) { if constexpr (true) x[0] = 1; }",
        "typedef float row_t[4];",
        "namespace N { typedef float row_t[4]; }",
        # pre-rtables-4a8f23c 1: the constexpr of a braceless if constexpr
        "float f(const thread float* x) { if constexpr (true) return x[0]; return 0; }",
        "struct S { float x[4]; };"
        " void f(thread S& s) { if constexpr (true) s.x[0] = 1; }",
        "void f(thread S* p) { if constexpr (true) p->x[0] = 1; }",
        "void f(thread float* x) { if constexpr (true) ++x[0]; }",
        "void f(thread float* x, thread float* y)"
        " { if constexpr (true) x[0] = 1, y[0] = 2; }",
        # pre-rtables-4a8f23c 2: structured bindings, array new and delete
        "auto [lo, hi] = g();",
        "void f() { static auto [lo, hi] = g(); }",
        "namespace N { constant auto [lo, hi] = p; }",
        "void* operator new[](size_t n);",
        "void operator delete[](void* p) noexcept;",
        # pre-rtables-9ff8169 1: '!=' in a template method does not leak
        "struct F { static metal::conditional_t<K != 0, float, half> make()"
        " { return 0; } float a[4]; };",
        "struct F { constexpr metal::enable_if_t<K != 0, float> make() const"
        " { return 0; } float a[4]; };",
        # pre-rtables-9ff8169 2: cv-qualified structured bindings
        "auto const [lo, hi] = g();",
        "void f() { static auto const [lo, hi] = g(); }",
        "namespace N { auto constant [lo, hi] = p; }",
        # a linkage declaration is not a namespace block
        'extern "C" void f() { float a[4]; }',
        # a struct body in a namespace is not namespace scope for its members
        "namespace N { struct F { float a[4]; }; }",
    ],
)
def test_a_face_never_refuses_code_that_is_not_a_table(source):
    Dot("valid", source, "F", ())


@pytest.mark.parametrize("keyword", ["else", "do", "try"])
def test_the_scan_ends_else_do_and_try_bodies(keyword):
    assert not kernels._continues([keyword])


@pytest.mark.parametrize(
    "source",
    [
        # 2: parenthesized declarators
        "constant float (LUT[4]) = {1, 2, 3, 4};",
        "constant float (LUT)[4] = {1, 2, 3, 4};",
        "void f() { static constexpr float (LUT)[4] = {1}; }",
        "constant float (*LUT[4]) = {nullptr};",
        "constant int (*LUT[4]);",
        "constant int (*LUT[4])(int);",
        "void f() { static int (*LUT[4])(int); }",
        # 5: variable template specializations
        "template <> constant float LUT<int>[4] = {1};",
        "template <int K> constant float LUT<K, 0>[4] = {1};",
    ],
)
def test_exotic_spellings_are_documented_as_not_detected(source):
    # Such a table stays in constant memory: it costs speed, results are the
    # same. The guide and the refusal say so, and review refuses it.
    assert kernels.hand_tables(source, face=True) == ()
    Dot("unread", source, "F", ())


def test_the_guide_and_the_refusal_name_what_the_scan_does_not_read():
    page = (ROOT / "docs" / "quant-formats.md").read_text()
    section = page.split("\n## Tables\n", 1)[1].split("\n## ", 1)[0]
    with pytest.raises(ValueError) as refused:
        Dot("said", "constant float LUT[4] = {1};", "F", ())
    guide = " ".join(section.split())
    assert "#if 0" in guide
    assert "brace-initialized constructor" in guide
    for said in (guide, str(refused.value)):
        assert "parenthesized declarator" in said
        assert "variable template specialization" in said
        assert "only costs speed" in said
    assert "cannot classify" not in page


@pytest.mark.parametrize("part", ["source", "shared"])
def test_a_face_requires_metal_text_in_each_fragment(part):
    args = {"key": "bad", "source": "struct F {};", "type": "F", "tensors": ()}
    with pytest.raises(ValueError, match=rf"face 'bad': {part} is Metal"):
        Dot(**{**args, part: 7})


def test_a_face_can_initialize_an_automatic_aggregate():
    Dot("automatic", "struct S { int a; }; void f() { S s{0}; }", "F", ())


def test_a_shared_source_sees_types_declared_by_the_main_source():
    with pytest.raises(ValueError, match=r"line 1 of its shared source declares LUT"):
        Dot(
            "split",
            "struct S { float v[4]; };",
            "Face",
            (),
            shared="constant S LUT[2];",
            vpt=8,
        )


def test_faces_share_the_declaration_scan_for_equal_source():
    scan = kernels.hand_tables
    source = "constant float LUT[7];"
    scan.cache_clear()
    try:
        first = scan(source)
        assert scan((" " + source)[1:]) == first
        info = scan.cache_info()
        assert (info.hits, info.misses) == (1, 1)
    finally:
        scan.cache_clear()


def test_a_later_declarator_reports_its_own_line():
    text = "constant int S = 1,\n    LUT[16] = {1};"
    assert hand_tables(text) == ((2, "LUT", "LUT[16] = {1};"),)


def test_a_table_is_found_on_its_line():
    source = (
        "namespace a {\nconstant constexpr int B = 4;\n\n"
        "constant float L[2] = {1, 2};\n}"
    )
    assert hand_tables(source) == ((4, "L", "constant float L[2] = {1, 2};"),)


_HAND = r"""
namespace omlx_hand {
constant constexpr float LEVELS[] = {-1.0f, 1.0f};
struct Dot {
  static constant constexpr int CONTRACT = 1;
};
}
"""


def test_an_implicit_static_table_names_its_storage_rule():
    with pytest.raises(ValueError, match=r"LUT with static storage"):
        Dot("implicit", "namespace N { constexpr float LUT[4] = {1}; }", "N::Dot", ())


def test_a_face_that_declares_a_table_by_hand_is_refused_when_it_is_made():
    with pytest.raises(ValueError) as refused:
        Dot(key="hand", source=_HAND, type="omlx_hand::Dot", tensors=())
    said = str(refused.value)
    assert said.startswith(
        "quant format face 'hand': line 3 of its source declares LEVELS"
    )
    assert "constant constexpr float LEVELS[] = {-1.0f, 1.0f};" in said
    assert "Dot(..., tables=(Table(name, dtype, values),))" in said
    assert "omlx_table" in said
    assert "docs/quant-formats.md, Tables" in said
    # Declared as data, the same face is made.
    source = _HAND.replace("constant constexpr float LEVELS[] = {-1.0f, 1.0f};\n", "")
    Dot(key="hand", source=source, type="omlx_hand::Dot", tensors=(), tables=(LUT,))


class _Hand(formats.QuantFormat):
    mode, tensors, contract = "hand", ("scales",), formats.CONTRACT

    def dot(self, layer, k, n):
        return Dot(key="hand", source=_HAND, type="omlx_hand::Dot", tensors=("scales",))

    def layout(self, k, n, bits, group_size):
        return {"weight": ((n, k), mx.uint8), "scales": ((n,), mx.float32)}

    def encode(self, w, importance, bits, group_size):
        return {"weight": mx.zeros(w.shape, mx.uint8), "scales": mx.ones(w.shape[:-1])}


@testing.verified("llama")
def test_a_model_runs_such_a_format_s_layers_without_the_face(monkeypatch, dot_defects):
    formats.load()
    monkeypatch.setitem(formats._FORMATS, "hand", _Hand())
    layer = format_linear.FormatLinear(
        "hand",
        8,
        1,
        64,
        {"weight": mx.zeros((16, 64), mx.uint8), "scales": mx.ones((16,))},
    )
    assert formats.dot(layer, 64, 16) is None
    ((mode, error),) = dot_defects
    assert mode == "hand" and "declares LEVELS with static storage" in error
    dot_defects.clear()  # expected


def test_no_kernel_built_over_faces_holds_a_table_written_by_hand(monkeypatch):
    """The kernels over faces, with faces of every kind of table and without:
    what they compile holds no table but the staged ones, in oMLX's own
    section."""
    staged = re.compile(r'#line 1 "<omlx staged tables>"\n.*?(?=\n#line |\Z)', re.S)
    captures = []

    def record(**kwargs):
        text = kwargs.get("header", "") + "\n" + kwargs["source"]
        assert not hand_tables(staged.sub("", text)), kwargs["name"]
        captures.append(bool(staged.search(text)))

    monkeypatch.setattr(mx.fast, "metal_kernel", record)
    monkeypatch.setattr(kernels, "_FACE_KERNELS", {})
    _capture_nax_without_mlx_headers(monkeypatch)
    plain = affine.face(4, 64, True, mx.bfloat16)
    shared = affine.face(8, 128, True, mx.bfloat16)
    gate = affine.face(8, 64, True, mx.bfloat16)
    for face in (plain, tabled(plain)):
        for stacked in (False, True):
            format_linear._kernel.__wrapped__(face, stacked)
        gather._kernel.__wrapped__(face)
        gather._ordered_kernel.__wrapped__(face)
        for variant in (False, True):
            testing._dot_rows_kernel.__wrapped__(face, variant)
            testing._shared_rows_kernel.__wrapped__(face, variant)
            testing._lane_parts_kernel.__wrapped__(face, variant)
        testing._finish_rows_kernel.__wrapped__(face)
        kernels.face_kernel("scan", ("m",), {"m": face}, ("out",), "", "")
        for builder in (routed._gate_up_kernel, routed._gate_up_topk_kernel):
            builder.__wrapped__(face, None, None)
            builder.__wrapped__(face, tabled(plain), gate)
        routed._down_kernel.__wrapped__(face, None)
        routed._down_kernel.__wrapped__(face, shared)
        routed._gate_up_window_kernel.__wrapped__(face, shared, gate)
        routed._down_window_kernel.__wrapped__(face, shared)
        table = bool(face.tables)
        assert captures == [table] * 12 + [table, True] * 2 + [table] * 4
        captures.clear()
    toy = _toy(
        "every",
        "  omlx_table<half, 2> a;\n  omlx_table<int8_t, 3> b;",
        Table("a", mx.float16, (1.0, 2.0)),
        Table("b", mx.int8, (1, 2, 3)),
    )
    testing._staged_tables_kernel.__wrapped__(toy)
    tile = affine.tile_face(4, 64, mx.bfloat16)
    for face in (tile, tabled_tile(tile)):
        testing._tile_rows_kernel.__wrapped__(face)
        for kind in ("plain", "act", "act_map"):
            nax._face_kernel(face, kind)
    assert captures == [True] + [False] * 4 + [True] * 4


def test_served_affine_sources_only_declare_the_formats_tables(monkeypatch):
    # Keep the same inventory as the byte-for-byte source pin, including new
    # builders as they are added to it. Scan the header and kernel body apart.
    from tests.test_formats_tables import _affine_sources, _steel_sources

    records = _affine_sources(monkeypatch) + _steel_sources(monkeypatch)
    for name, header, body in records:
        assert not hand_tables(header), name
        assert not hand_tables(body), name
